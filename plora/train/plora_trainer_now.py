from transformers import Trainer
import numpy as np
from transformers.trainer import (
    is_torch_tpu_available,
    Dict,
    os,
    torch,
    nn,
    deepspeed_init,
    ShardedDDPOption,
    is_sagemaker_mp_enabled,
    math,
    logger,
    skip_first_batches,
    has_length,
    sys,
    TrainerState,
    time,
    DebugOption,
    version,
    accelerate_version,
    dist,
    speed_metrics,
    get_model_param_count,
    DebugUnderflowOverflow,
    deepspeed_load_checkpoint,
    TRAINER_STATE_NAME,
    HPSearchBackend,
    hp_params,
    ParallelMode,
    shutil,
    TrainOutput
)
import ipdb

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0.1, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    
    
    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)




class pl_trainer(Trainer):
    def __init__(
        self,
        model_name = 'flan',
        use_deepspeed = True,
        is_test = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # self.model_name = kwargs.get('args').model_name_or_path
        self.use_deepspeed = use_deepspeed
        self.is_test = is_test
        self.init_interval = kwargs.get('args').init_interval
        self.interval = 0
        self.update_op = kwargs.get('args').update_op
        self.update_op_cheng = kwargs.get('args').update_op_cheng
        self.pmr_op = kwargs.get('args').pmr_op
        self.update = 0
        self.update = self.init_interval
        self.decay = kwargs.get('args').decay
        self.lora_offload = 0
        self.auto_update_es = False
        self.auto_update_time = False
        self.es_patience = kwargs.get('args').es_patience
        self.norm_update_interval = kwargs.get('args').norm_update_interval 
        self.min_delta = kwargs.get('args').min_delta
        self.plora_momentum = kwargs.get('args').plora_momentum
        self.plora_momentum_ratio = kwargs.get('args').plora_momentum_ratio
        self.save_flag = kwargs.get('args').save_flag
        self.custom_save_interval = kwargs.get('args').custom_save_interval
        self.update_num = kwargs.get('args').update_num
        self.perdic_flag = kwargs.get('args').perdic_flag
        self.custom_eval_interval = kwargs.get('args').custom_eval_interval
        self.eval_flag = kwargs.get('args').eval_flag
        self.update_based_rank = kwargs.get('args').update_based_rank
        self.update_based_rank_time = False
        self.update_based_rank_interval = kwargs.get('args').update_based_rank_interval
        self.update_based_eval_loss =kwargs.get('args').update_based_eval_loss
        self.update_based_eval_loss_interval =kwargs.get('args').update_based_eval_loss_interval

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        # print('######################进入私有_maybe_log_save_evaluate')
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
                    metrics.update(dataset_metrics)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            if model.device.index == 0:
                print('######################保存模型')
                save_id = f"{self.state.global_step}-savemodel.pth"
                savemodel_dir = os.path.join(self.args.output_dir, save_id)
                torch.save(model.state_dict(), savemodel_dir)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control) 

   
        return metrics

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps and args.logging_steps < 1:
            args.logging_steps = math.ceil(max_steps * args.logging_steps)
        if args.eval_steps and args.eval_steps < 1:
            args.eval_steps = math.ceil(max_steps * args.eval_steps)
        if args.save_steps and args.save_steps < 1:
            args.save_steps = math.ceil(max_steps * args.save_steps)
        eval_loss_best=1000.0
        if_update_base_evalloss_time=False
        no_improvement_count=0
        eval_loss=1000.0
        improved=False
        
        
        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
        )
        ##############modify
        steps_every = max_steps // num_train_epochs
        global_steps_gate = steps_every
        if self.update_op_cheng == 1 and self.update_op == 0:
            
            if self.auto_update_es:
                self.update = float('inf')
            
            elif self.update_num!=0:
                self.interval = steps_every // self.update_num
                self.update = self.interval
            else:
                self.update = self.init_interval
        elif self.update_op_cheng != 1:
            self.update = self.init_interval
            self.interval = self.init_interval * self.update_op_cheng
        elif self.update_op != 0:
            self.update = self.init_interval
            self.interval = self.init_interval + self.update_op_cheng
        else:
            print('######################error!!self.update_interval',self.update_op_cheng)
        if self.update_based_eval_loss:
            self.update = 10000000
            self.update_op_cheng =1 
            self.update_op = 0
        ranks=[[] for _ in range(230)] 
        norms=[[] for _ in range(230)] 
        denorms=[[] for _ in range(230)] 

        base_pre_weight=[0.0 for _ in range(50)] 
        j=0
        

        
        ##############modify
        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)
        # print('######################初始化的optimizer',self.optimizer)
        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # Fairscale Sharded DDP, FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        for i in [0,5,10,15,20,25,31]:
            for mo,adapter in zip(["self_attn","self_attn","self_attn","self_attn","mlp","mlp","mlp"],['k_proj', 'q_proj', 'o_proj', 'v_proj', 'up_proj', 'gate_proj', 'down_proj']):
                base_pre_weight[j] = model.base_model.base_model.layers[i].__getattr__(mo).__getattr__(adapter).weight.clone().detach().cpu()
                j+=1

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # deepspeed ckpt loading
        if resume_from_checkpoint is not None and self.is_deepspeed_enabled:
            deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                for _ in train_dataloader:
                    break

        total_batched_samples = 0
        
        early_stop = EarlyStopping(min_delta=self.min_delta,patience=self.es_patience)

       
        if self.update_based_rank == True:
            self.plora_momentum = False  
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True
            step = -1
            # continue_training = True
            updatei=0
            for step, inputs in enumerate(epoch_iterator):

                total_batched_samples += 1
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                # if early_stop.step(tr_loss):
                #     self.auto_update_es=True

                if self.auto_update_es:
                    if self.state.global_step % self.norm_update_interval == 0:
                        for i in range(len(self.model.base_model.model.model.layers)):
                                        for adapter in self.model.peft_config['default'].target_modules:
                                            if adapter in ['k_proj', 'q_proj', 'o_proj', 'v_proj']:
                                                for item in [self.model.base_model.model.model.layers[i].self_attn.__getattr__(adapter).lora_A['default'], self.model.base_model.model.model.layers[i].mlp.__getattr__(adapter).lora_A['default'],model.base_model.model.model.layers[i].self_attn.__getattr__(adapter).lora_B['default'],model.base_model.model.model.layers[i].mlp.__getattr__(adapter).lora_B['default']]:
                                                    model_mo = torch.norm(item.weight, p=2)
                                                    if  early_stop.step(model_mo):
                                                        pass
                                                        # self.auto_update_es=True


                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc or (
                        version.parse(accelerate_version) <= version.parse("0.20.3")
                    ):
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        if self.do_grad_scaling:
                            # Reduce gradients first for XLA
                            if is_torch_tpu_available():
                                gradients = xm._fetch_gradients(self.optimizer)
                                xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if is_torch_tpu_available():
                        if self.do_grad_scaling:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            # tpu-comment: accelerate wrapped optimizers call xm.optimizer_step
                            self.optimizer.step()
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()
                        optimizer_was_run = not self.accelerator.optimizer_step_was_skipped

                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    #======================modify==============================
                    ###如果是根据rank自动更新的话，由update来赋值测试间隔
                        
                    if self.state.global_step>=self.update or if_update_base_evalloss_time:
                       
                        decay_rate = self.decay**(self.lora_offload)
                        self.lora_offload+=1
                        
                        if self.lora_offload % self.custom_save_interval == 0:
                            if self.save_flag:
                                self.control.should_save = True
                        

                         ##############modify
                        if self.update_op_cheng == 1 and self.update_op == 0:
                            self.update+=self.interval
                            if self.update_num != 1:
                                if self.update+self.interval > global_steps_gate:
                                    self.update=global_steps_gate
                                    global_steps_gate = global_steps_gate + steps_every
                            elif self.update_num == 1:
                                global_steps_gate = global_steps_gate + steps_every
                        else:    
                            if self.update_op_cheng != 1:
                                self.interval *= self.update_op_cheng
                                self.next_interval = self.interval * self.update_op_cheng
                            elif self.update_op != 0:
                                self.interval += self.update_op        
                                self.next_interval = self.interval + self.update_op                                          
                            self.update += self.interval
                            if self.update + self.next_interval > max_steps:
                                self.update = max_steps
                        ##############modify                        
                                        
                        if self.perdic_flag :                   
                            if 'llama' in self.args.output_dir:
                                
                                if_update_base_rank_time =False
                                index=0
                                indexr=0
                                for i in range(len(model.base_model.model.model.layers)):                                    
                                    for mo,adapter in zip(["self_attn","self_attn","self_attn","self_attn","mlp","mlp","mlp"],['k_proj', 'q_proj', 'o_proj', 'v_proj', 'up_proj', 'gate_proj', 'down_proj']):
                                        lora_a = model.base_model.model.model.layers[i].__getattr__(mo).__getattr__(adapter).lora_A['default']
                                        lora_b = model.base_model.model.model.layers[i].__getattr__(mo).__getattr__(adapter).lora_B['default']
                                        if self.update_based_rank == True:
                                            lora_ab = lora_b.weight @ lora_a.weight
                                            r = lora_a.weight.shape[0]
                                            lora_ab_float = lora_ab.to(torch.float32)
                                            _, S, _ = torch.svd(lora_ab_float)                                                                                        
                                                                                                                                                
                                            rank = torch.sum(S > 5e-2).item()                                                
                                            ranks[indexr].append(rank)
                                            norms[indexr].append(torch.norm(model.base_model.model.model.layers[i].__getattr__(mo).__getattr__(adapter).base_layer.weight, p=2).item())
                                            indexr+=1

                                            if i in [0,5,10,15,20,25,31]:
                                                basew=base_pre_weight[index].cuda()
                                                
                                                denorms[index].append(torch.norm(model.base_model.model.model.layers[i].__getattr__(mo).__getattr__(adapter).base_layer.weight - basew, p=2).item())
                                                index+=1
                                                del basew
                                                                                        
                                            if rank >= r :
                                                if_update_base_rank_time = True
                                            
                                        base = model.base_model.model.model.layers[i].__getattr__(mo).__getattr__(adapter).base_layer
                                        
                                        # base.weight.mul_(1-decay_rate * model.peft_config['default'].lora_alpha)
                                        scaling = model.peft_config['default'].lora_alpha / model.peft_config['default'].r
                                        if args.plora_momentum or if_update_base_rank_time or if_update_base_evalloss_time:
                                            delta_ab = lora_b.weight @ lora_a.weight * scaling * self.plora_momentum_ratio * decay_rate                                      
                                            if_update_base_rank = False
                                            with torch.no_grad():
                                                updatei+=1                                  
                                                model.base_model.model.model.layers[i].__getattr__(mo).__getattr__(adapter).base_layer.weight.copy_(delta_ab + base.weight)
                                                model.base_model.model.model.layers[i].__getattr__(mo).__getattr__(adapter).lora_A['default'].weight.mul_(math.sqrt(1-self.plora_momentum_ratio))
                                                model.base_model.model.model.layers[i].__getattr__(mo).__getattr__(adapter).lora_B['default'].weight.mul_(math.sqrt(1-self.plora_momentum_ratio))
                                            
                                            
                                        else:
                                         
                                            f_out, f_in = base.weight.shape 
                                            r = lora_a.weight.shape[0]
                                            delta_ab = lora_b.weight @ lora_a.weight * scaling * decay_rate

                                            with torch.no_grad():
                                                model.base_model.model.model.layers[i].__getattr__(mo).__getattr__(adapter).base_layer.weight.copy_(delta_ab + base.weight)
                                                model.base_model.model.model.layers[i].__getattr__(mo).__getattr__(adapter).lora_A['default'].weight.copy_(nn.Linear(f_in, r, bias=False).weight) 
                                                nn.init.kaiming_uniform_(model.base_model.model.model.layers[i].__getattr__(mo).__getattr__(adapter).lora_A['default'].weight, a=math.sqrt(5))
                                                nn.init.zeros_(model.base_model.model.model.layers[i].__getattr__(mo).__getattr__(adapter).lora_B['default'].weight)

                                         
                                for blocki in model.base_model.model.encoder.block:
                                    
                                    for adapter in model.peft_config['default'].target_modules:
                                        if adapter in ['k', 'q', 'o', 'v']:
                                            layerN=0
                                            mname="SelfAttention"
                                        else:
                                            layerN=1
                                            mname="DenseReluDense"
                                        
                                        lora_a = blocki.layer[layerN].__getattr__(mname).__getattr__(adapter).lora_A['default']
                                        lora_b = blocki.layer[layerN].__getattr__(mname).__getattr__(adapter).lora_B['default']
                                        base = blocki.layer[layerN].__getattr__(mname).__getattr__(adapter).base_layer
                                        # ipdb.set_trace()
                                        scaling = model.peft_config['default'].lora_alpha / model.peft_config['default'].r
                                        if args.plora_momentum:
                                            # print('#######################plora_momentum更新update--------')
                                            delta_ab = lora_b.weight @ lora_a.weight * scaling * self.plora_momentum_ratio * decay_rate

                                            with torch.no_grad():
                                                blocki.layer[layerN].__getattr__(mname).__getattr__(adapter).base_layer.weight.copy_(delta_ab + base.weight)
                                                blocki.layer[layerN].__getattr__(mname).__getattr__(adapter).lora_A['default'].weight.mul_(math.sqrt(1-self.plora_momentum_ratio))
                                                blocki.layer[layerN].__getattr__(mname).__getattr__(adapter).lora_B['default'].weight.mul_(math.sqrt(1-self.plora_momentum_ratio))
                                        else:
                                        # print('#######################plora_no_momentum')
                                            f_out, f_in = base.weight.shape 
                                            r = lora_a.weight.shape[0]
                                            delta_ab = lora_b.weight @ lora_a.weight * scaling * decay_rate

                                            with torch.no_grad():
                                                blocki.layer[layerN].__getattr__(mname).__getattr__(adapter).base_layer.weight.copy_(delta_ab + base.weight)
                                                blocki.layer[layerN].__getattr__(mname).__getattr__(adapter).lora_A['default'].weight.copy_(nn.Linear(f_in, r, bias=False).weight) 
                                                nn.init.kaiming_uniform_(blocki.layer[layerN].__getattr__(mname).__getattr__(adapter).lora_A['default'].weight, a=math.sqrt(5))
                                                nn.init.zeros_(blocki.layer[layerN].__getattr__(mname).__getattr__(adapter).lora_B['default'].weight)

                            self.plora_momentum_ratio += self.pmr_op              
                            if self.plora_momentum_ratio >1:
                                self.plora_momentum_ratio=1
                            if self.plora_momentum_ratio <0:
                                self.plora_momentum_ratio=0.001
                            # RESET OPTIMIZER
                            def random_pruning_(tensor, prune_ratio):
                                """
                                Performs random pruning dimensionality reduction **inplace**.
                                Only reduces the inner dimensionality, does not affect the shape of the tensor
                                """
                                random_pruning_mask = torch.rand_like(tensor) > prune_ratio
                                tensor.mul_(random_pruning_mask)
                            from functools import partial
                            pruning_fn = partial(random_pruning_, prune_ratio=0.999)
                            optimizer_state_keys = ["exp_avg", "exp_avg_sq"]

                            for name in optimizer_state_keys:
                                pruning_fn(self.optimizer.state[set(self.optimizer.state).pop()][name])
                                # print('#######################optimizer',self.optimizer.state[set(self.optimizer.state).pop()][name])
                        else:
                            print('#######################no_peridic')

                    #======================modify==============================

                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    # self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                    #eval_modified
                    if self.state.global_step % self.update_based_eval_loss_interval == 0:
                        if self.eval_flag:
                            self.control.should_evaluate = True

                    if self.control.should_evaluate:
                        eval_metric=self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                        eval_loss=eval_metric["eval_loss"]
                    else:
                        eval_metric=self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                    
                    if self.state.global_step % self.update_based_eval_loss_interval == 0 and self.update_based_eval_loss:
                        if no_improvement_count >= 2: #eval loss作为指标不一定准确。这个阈值是不是需要。 会不会跳出局部最优
                            if not improved:
                                #break 要不就不再update，要不就early-stop
                                self.perdic_flag = False
                                self.update_based_eval_loss = False
                                
                            else:
                                #update todo
                                if_update_base_evalloss_time = True
                                print("update---------------")
                                improved = False
                                no_improvement_count = 0

                        if eval_loss < eval_loss_best:
                            eval_loss_best = eval_loss
                            improved = True
                            no_improvement_count = 0
                        else:
                            no_improvement_count += 1

              
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.state.global_step % 25 == 0:                         
                    index=0                           
                    for i in [0,5,10,15,20,25,31]:
                        for mo,adapter in zip(["self_attn","self_attn","self_attn","self_attn","mlp","mlp","mlp"],['k_proj', 'q_proj', 'o_proj', 'v_proj', 'up_proj', 'gate_proj', 'down_proj']):  
                            now = model.base_model.model.model.layers[i].self_attn.__getattr__(adapter).base_layer.weight
                            basew=base_pre_weight[index].cuda()
                            cha = now - base_pre_weight[index]
                            del basew
                            norms[index].append(torch.norm(now, p=2).item())
                            denorms[index].append(torch.norm(cha, p=2).item())                                            

                            lora_ab_float = cha.to(torch.float32)
                            _, S, _ = torch.svd(lora_ab_float)
                            rank = torch.sum(S > 5e-2).item()                                        
                            ranks[index].append(rank)                           
                            index+=1

                
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            # self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
            # modified
            eval_metric = self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)                  
            # eval_loss = eval_metric["eval_loss"]
            
            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)
        print('#######################plora_momentum_update次数',updatei)
                            
        print("--------------------self.plora_momentum_ratio* lr",self.plora_momentum_ratio * self._get_learning_rate())
        self.log(metrics)

        l=len(ranks[0])
        import csv
        with open(f"/mnt/data/wxc/workspace/project/PLoRA/results_log/ranks_{pi}.csv",mode='w',newline='') as file:
            writer=csv.writer(file)
            writer.writerows(ranks) 

        with open(f"/mnt/data/wxc/workspace/project/PLoRA/results_log/normW_{pi}.csv",mode='w',newline='') as file:
            writer=csv.writer(file)
            writer.writerows(norms)             
        with open(f"/mnt/data/wxc/workspace/project/PLoRA/results_log/normDeltaW_{pi}.csv",mode='w',newline='') as file:
            writer=csv.writer(file)
            writer.writerows(denorms)   

        import matplotlib.pyplot as plt
        import matplotlib.pyplot as plt1
        import matplotlib.pyplot as plt2
        import matplotlib.pyplot as plt3

        # 使用 matplotlib 创建折线图
        plt.figure(figsize=(10, 5))  # 图形的大小
        plt1.figure(figsize=(10, 5))
        plt2.figure(figsize=(10, 5))
        plt3.figure(figsize=(10, 5))

        index=0
        pi=os.path.basename(args.output_dir)
        for i in [0,5,10,15,20,25,31]:
            for mo,adapter in zip(["self_attn","self_attn","self_attn","self_attn","mlp","mlp","mlp"],['k_proj', 'q_proj', 'o_proj', 'v_proj', 'up_proj', 'gate_proj', 'down_proj']): 

                plt.plot(ranks[index], marker='o', linestyle='-', label=str(i)+str(adapter))
                plt1.plot(ranks[index], marker='o', linestyle='-', label=str(i)+str(adapter))
                plt2.plot(ranks[index], marker='o', linestyle='-', label=str(i)+str(adapter))
                index+=1

        plt.title('all-ranks')  # 设置图形的标题
        plt.xlabel('Index')  # 设置x轴标题
        plt.ylabel('rank')  # 设置y轴标题
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1)  # 显示图例
        plt.subplots_adjust(right=0.85) 
        plt.grid(True)  # 显示网格                 
        plt.savefig(f'/mnt/data/wxc/workspace/project/PLoRA/results_log/all{pi}_plot.png')

        plt1.title('all-norms')  # 设置图形的标题
        plt1.xlabel('Index')  # 设置x轴标题
        plt1.ylabel('norm')  # 设置y轴标题
        plt1.legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1)  # 显示图例
        plt1.subplots_adjust(right=0.85) 
        plt1.grid(True)  # 显示网格                 
        plt1.savefig(f'/mnt/data/wxc/workspace/project/PLoRA/results_log/norms{pi}_plot.png')

        plt2.title('all-delta-norms')  # 设置图形的标题
        plt2.xlabel('Index')  # 设置x轴标题
        plt2.ylabel('dnorm')  # 设置y轴标题
        plt2.legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1)  # 显示图例
        plt2.subplots_adjust(right=0.85) 
        plt2.grid(True)  # 显示网格                 
        plt2.savefig(f'/mnt/data/wxc/workspace/project/PLoRA/results_log/denorms{pi}_plot.png')

        index=0
        for i in [0,5,10,15,20,25,31]:
            for mo,adapter in zip(["mlp"],['up_proj']): 
                plt3.plot(ranks[index*4], marker='o', linestyle='-', label=str(i)+str(adapter))
                index+=7
        plt3.title('upproj-rank')  # 设置图形的标题
        plt3.xlabel('Index')  # 设置x轴标题
        plt3.ylabel('rank')  # 设置y轴标题
        plt3.legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1)  # 显示图例
        plt3.subplots_adjust(right=0.85) 
        plt3.grid(True)  # 显示网格                 
        plt3.savefig(f'/mnt/data/wxc/workspace/project/PLoRA/results_log/up-proj-rank{pi}_plot.png')

           

        
        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def offload_ab(self, model):
        # default deepspeed engine
        scaling = model.peft_config['default'].lora_alpha / model.peft_config['default'].r
        no_deepspeed = model.module
        lora_a_list = [key for key, _ in model.module.named_modules() if 'lora_A.default' in key]
        lora_b_list = [key for key, _ in model.module.named_modules() if 'lora_B.default' in key]
        base_list = [key for key, _ in model.module.named_modules() if 'base_layer' in key]
        for i in range(len(base_list)):
            lora_a = model.module.get_submodule(lora_a_list[i])
            lora_b = model.module.get_submodule(lora_b_list[i])
            base = model.module.get_submodule(base_list[i])
            delta_ab = lora_b.weight @ lora_a.weight * scaling * self.plora_momentum_ratio
            with torch.no_grad():
                if delta_ab.shape == base.weight.shape:
                    base.weight.copy_(base.weight + delta_ab)
                else:
                    delta_ab = delta_ab.t()
                    base.weight.copy_(base.weight + delta_ab)
                lora_a.weight.mul_(math.sqrt(1-self.plora_momentum_ratio))
                lora_b.weight.mul_(math.sqrt(1-self.plora_momentum_ratio))

class lora_trainer(Trainer):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.init_interval = kwargs.get('args').init_interval
        self.interval = kwargs.get('args').interval
        self.update = 0
        self.update = self.init_interval
        self.decay = 0.9
        self.lora_offload = 0
        self.save_flag = kwargs.get('args').save_flag
        self.custom_save_interval = kwargs.get('args').custom_save_interval
        # self.custom_eval_interval = kwargs.get('args').custom_eval_interval
        # self.eval_flag = kwargs.get('args').eval_flag

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps and args.logging_steps < 1:
            args.logging_steps = math.ceil(max_steps * args.logging_steps)
        if args.eval_steps and args.eval_steps < 1:
            args.eval_steps = math.ceil(max_steps * args.eval_steps)
        if args.save_steps and args.save_steps < 1:
            args.save_steps = math.ceil(max_steps * args.save_steps)

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
        )
        ##############modify
        steps_every = max_steps // num_train_epochs
        global_steps_gate = steps_every
        ##############modify
        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # Fairscale Sharded DDP, FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # deepspeed ckpt loading
        if resume_from_checkpoint is not None and self.is_deepspeed_enabled:
            deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                for _ in train_dataloader:
                    break

        total_batched_samples = 0
        # delta=[[0.0 for i in range(50)] for _ in range(50)] 
        ranks=[[] for _ in range(50)] 
        norms=[[] for _ in range(50)] 
        denorms=[[] for _ in range(50)] 

        if args.sft:
            base_pre_weight=[0.0 for _ in range(50)] 
            j=0
            for i in [0,5,10,15,20,25,31]:
                for mo,adapter in zip(["self_attn","self_attn","self_attn","self_attn","mlp","mlp","mlp"],['k_proj', 'q_proj', 'o_proj', 'v_proj', 'up_proj', 'gate_proj', 'down_proj']):
                    base_pre_weight[j] = model.base_model.base_model.layers[i].__getattr__(mo).__getattr__(adapter).weight.clone().detach().cpu()
                    j+=1
                    

        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
                ###modifi  debug!!!!!!
                    if step % 25 == 0:
                        if not args.sft: 
                            index=0                           
                            for i in [0,5,10,15,20,25,31]:
                                for mo,adapter in zip(["self_attn","self_attn","self_attn","self_attn","mlp","mlp","mlp"],['k_proj', 'q_proj', 'o_proj', 'v_proj', 'up_proj', 'gate_proj', 'down_proj']):  
                                    lora_a = model.base_model.model.model.layers[i].__getattr__(mo).__getattr__(adapter).lora_A['default'].clone().detach().cpu()
                                    lora_b = model.base_model.model.model.layers[i].__getattr__(mo).__getattr__(adapter).lora_B['default'].clone().detach().cpu()
                                    lora_ab = lora_b.weight @ lora_a.weight
                                    lora_ab=self._get_learning_rate() * lora_ab
                                    r = lora_a.weight.shape[0]
                                    lora_ab_float = lora_ab.to(torch.float32)
                                    _, S, _ = torch.svd(lora_ab_float)
                                    rank = torch.sum(S > 5e-2).item()
                                    
                                    ranks[index].append(rank) 
                                    norms[index].append(torch.norm(model.base_model.base_model.layers[i].__getattr__(mo).__getattr__(adapter).weight,2).item())
                                    denorms[index].append(torch.norm(delta,2).item())
                                    index+=1
                                    del lora_ab
                                      
                        else:#sft                            
                            index=0
                            for i in [0,5,10,15,20,25,31]:
                                for mo,adapter in zip(["self_attn","self_attn","self_attn","self_attn","mlp","mlp","mlp"],['k_proj', 'q_proj', 'o_proj', 'v_proj', 'up_proj', 'gate_proj', 'down_proj']):
                                    
                                    # basew=base_pre_weight[index].cuda()
                                    delta=  model.base_model.base_model.layers[i].__getattr__(mo).__getattr__(adapter).weight.detach().cpu() - base_pre_weight[index]
                                    # del basew                                                                    
                                    
                                    lora_ab_float = delta.to(torch.float32)
                                    _, S, _ = torch.svd(lora_ab_float)                                    
                                    rank = torch.sum(S > 5e-2).item()
                                    ranks[index].append(rank)
                                    norms[index].append(torch.norm(model.base_model.base_model.layers[i].__getattr__(mo).__getattr__(adapter).weight,2).item())
                                    denorms[index].append(torch.norm(delta,2).item())
                                    index +=1
                                        
                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc or (
                        version.parse(accelerate_version) <= version.parse("0.20.3")
                    ):
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        if self.do_grad_scaling:
                            # Reduce gradients first for XLA
                            if is_torch_tpu_available():
                                gradients = xm._fetch_gradients(self.optimizer)
                                xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if is_torch_tpu_available():
                        if self.do_grad_scaling:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            # tpu-comment: accelerate wrapped optimizers call xm.optimizer_step
                            self.optimizer.step()
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()
                        optimizer_was_run = not self.accelerator.optimizer_step_was_skipped

                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    #==============================modify====================================
                                       
                    if self.state.global_step>=self.update:
                        if model.device.index==0:
                            print("自动的更新",self.state.global_step>=self.update, self.lora_offload)                       
                            print("global_step>=update_step", self.state.global_step,'/',self.update)
                    
                        self.lora_offload+=1

                        if self.lora_offload % self.custom_save_interval==0:
                            if self.save_flag:
                                self.control.should_save = True
                        

                        self.update+=self.interval
                        # ipdb.set_trace()
                        #收纳最后
                        if self.update+self.interval > global_steps_gate:
                            self.update=global_steps_gate
                            global_steps_gate = global_steps_gate + steps_every

                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    if not args.sft:
                        self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        import matplotlib.pyplot as plt
        # 使用 matplotlib 创建折线图
        plt.figure(figsize=(10, 5))  # 图形的大小
        
        index=0
        for i in [0,5,10,15,20,25,31]:
            for mo,adapter in zip(["self_attn","self_attn","self_attn","self_attn","mlp","mlp","mlp"],['k_proj', 'q_proj', 'o_proj', 'v_proj', 'up_proj', 'gate_proj', 'down_proj']): 
      
            # 绘制第一个数组
                plt.plot(ranks[index], marker='o', linestyle='-', label=str(i)+str(adapter))
                index+=1


        plt.title('Line Graph of Three Arrays')  # 设置图形的标题
        plt.xlabel('Index')  # 设置x轴标题
        plt.ylabel('rank')  # 设置y轴标题
        plt.legend()  # 显示图例
        plt.grid(True)  # 显示网格
       
        # plt.show()  # 显示图形
        pi=os.path.basename(args.output_dir)
        plt.savefig(f'/mnt/data/wxc/workspace/project/PLoRA/results_log/{pi}_plot.png')

        import csv
        with open(f"/mnt/data/wxc/workspace/project/PLoRA/results_log/ranks_{pi}.csv",mode='w',newline='') as file:
            writer=csv.writer(file)
            writer.writerows(ranks)
        with open(f"/mnt/data/wxc/workspace/project/PLoRA/results_log/normW_{pi}.csv",mode='w',newline='') as file:
            writer=csv.writer(file)
            writer.writerows(norms)             
        with open(f"/mnt/data/wxc/workspace/project/PLoRA/results_log/normDeltaW_{pi}.csv",mode='w',newline='') as file:
            writer=csv.writer(file)
            writer.writerows(denorms)           

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        # print("#---------------self.auto_update_es:",self.lora_offload)
        return TrainOutput(self.state.global_step, train_loss, metrics)
