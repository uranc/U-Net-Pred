import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from model.model_fn import build_compile_model
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from model.tensorboard_custom import TensorBoardCustom
import logging
import os
import pickle
from tensorflow.keras import backend as K
import numpy as np
import time
from tensorflow.keras import callbacks as cb
from model.input_fn import train_inputs_fn, valid_inputs_fn
import pdb

logging.basicConfig(level=logging.DEBUG)


class KerasWorker(Worker):
    def __init__(self, mode, params, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.params = params
        self.save_dir = params['exp_dir']

    def compute(self, config, budget, working_directory, *args, **kwargs):

        # build
        K.clear_session()
        worker_params = self.params
        worker_params['inst_norm'] = False  # config['inst_norm']
        worker_params['p_lr'] = config['p_lr']
        worker_params['p_fft_log'] = config['p_fft_log']
        worker_params['p_fft_phase'] = config['p_fft_phase']
        worker_params['p_fft_c'] = config['p_fft_c']
        worker_params['p_fft_s'] = config['p_fft_s']

        # inputs
        train_inputs, train_size = train_inputs_fn(
            worker_params['b_size'], worker_params['n_epoch'])
        valid_inputs, valid_size = valid_inputs_fn(
            worker_params['b_size'], worker_params['n_epoch'])
        train_steps = int(np.floor(4096*4/worker_params['b_size']))
        valid_steps = int(np.floor(1024/worker_params['b_size']))
        # train_steps = 3
        # build model
        model = build_compile_model(self.mode, worker_params)

        # Get relevant graph operations or nodes needed for training
        images, masks, mtype = train_inputs.get_next()
        val_im, val_mask, val_type = valid_inputs.get_next()
        print
        # get fit
        hist = model.fit([images]+[masks], images,
                         steps_per_epoch=train_steps,
                         epochs=int(budget),
                         validation_data=(([val_im]+[val_mask], val_im)),
                         validation_steps=valid_steps,
                         verbose=1)
        # pdb.set_trace()
        return ({
                # 'loss': float(hist.history['ssim'][-1]),
                # 'loss': float(hist.history['val_loss'][-1]),
                'loss': float(hist.history['val_ssim'][-1]),
                # 'loss': hist.history['loss'][-1],  # HpBandSter minimizes!
                # 'loss': hist.history['val_ssim_loss'][-1],
                'info': {
                    'loss': float(hist.history['loss'][-1]),
                    'ssim': float(hist.history['ssim'][-1]),
                    'val_loss': float(hist.history['val_loss'][-1]),
                    'val_ssim': float(hist.history['val_ssim'][-1]),
                    # 'no_params': model.count_params(),
                    # 'epoch': hist.epoch,
                    # 'params': hist.params,
                    # 'logs': hist.history,
                }
                })

    @staticmethod
    def get_configspace():
        """
        It builds the configuration space with the needed hyperparameters.
        It is easily possible to implement different types of hyperparameters.
        Beside float-hyperparameters on a log scale,
        it is also able to handle categorical input parameter.
        :return: ConfigurationsSpace-Object
        """
        cs = CS.ConfigurationSpace()
        # inst_norm = CSH.UniformIntegerHyperparameter('inst_norm',
        #                                              lower=0,
        #                                              upper=1,
        #                                              default_value=0,
        #                                              log=False)
        # cs.add_hyperparameters([inst_norm])
        p_lr = CSH.UniformFloatHyperparameter('p_lr',
                                              lower=1e-6,
                                              upper=1,
                                              default_value='1e-1',
                                              log=True)
        cs.add_hyperparameters([p_lr])
        p_fft_log = CSH.UniformFloatHyperparameter('p_fft_log',
                                                   lower=1e-3,
                                                   upper=1e3,
                                                   default_value='0.5',
                                                   log=True)
        cs.add_hyperparameters([p_fft_log])
        p_fft_phase = CSH.UniformFloatHyperparameter('p_fft_phase',
                                                     lower=1e-5,
                                                     upper=1e3,
                                                     default_value='0.5',
                                                     log=True)
        cs.add_hyperparameters([p_fft_phase])
        p_fft_c = CSH.UniformFloatHyperparameter('p_fft_c',
                                                 lower=1e-3,
                                                 upper=1e3,
                                                 default_value='0.5',
                                                 log=True)
        cs.add_hyperparameters([p_fft_c])
        p_fft_s = CSH.UniformFloatHyperparameter('p_fft_s',
                                                 lower=1e-3,
                                                 upper=1e3,
                                                 default_value='0.5',
                                                 log=True)
        cs.add_hyperparameters([p_fft_s])
        return cs


# def evaluate_epoch(mode, params):
#     callbacks = []
#     callbacks.append(TensorBoardCustom(log_dir=params['exp_dir'],
#                                        histogram_freq=0,
#                                        write_graph=True,
#                                        write_images=True,
#                                        update_freq='epoch'))
#     model, input_size = build_compile_model(mode, params)
#     eval_steps = 1

#     # while True:
#     # yield K.get_session().run(next_batch)
#     return model.fit(steps_per_epoch=eval_steps,
#                      initial_epoch=params['current_epoch'],
#                      epochs=params['current_epoch']+1,
#                      verbose=1,
#                      callbacks=callbacks)


def farm_hyperparameters(mode,
                         params):

    # parameters
    min_budget = 8
    max_budget = 16
    n_iterations = 50
    shared_directory = params['exp_dir']+'/hyper'
    run_id = 'h0'

    # This example shows how to log live results. This is most useful
    # for really long runs, where intermediate results could already be
    # interesting. The core.result submodule contains the functionality to
    # read the two generated files (results.json and configs.json) and
    # create a Result object.

    # Start local worker
    if params['worker']:
        time.sleep(5)
        w = KerasWorker(mode,
                        params,
                        run_id=run_id,
                        nameserver='127.0.0.1',
                        timeout=120)
        w.run(background=False)
        exit(0)

    # Every process has to lookup the hostname
    NS = hpns.NameServer(run_id=run_id, host='127.0.0.1', port=None)
    ns_host, ns_port = NS.start()
    # NS.start()

    # Start local worker
    w = KerasWorker(mode,
                    params,
                    run_id=run_id,
                    nameserver='127.0.0.1',
                    timeout=120)
    w.run(background=True)
    # w.run(background=False)

    # Run an optimizer
    exists = os.path.isfile(shared_directory + '/prev' + '/results.json') and \
        os.path.isfile(shared_directory + '/prev' + '/configs.json')

    if exists:
        print('exists')
        previous_run = hpres.logged_results_to_HBS_result(
            shared_directory+'/prev')
        result_logger = hpres.json_result_logger(directory=shared_directory,
                                                 overwrite=False)

        # Run an optimizer
        bohb = BOHB(configspace=KerasWorker.get_configspace(),
                    run_id=run_id,
                    nameserver='127.0.0.1',
                    min_budget=min_budget,
                    max_budget=max_budget,
                    result_logger=result_logger,
                    previous_result=previous_run,
                    )

    else:
        # Run an optimizer
        result_logger = hpres.json_result_logger(directory=shared_directory,
                                                 overwrite=True)
        bohb = BOHB(configspace=KerasWorker.get_configspace(),
                    run_id=run_id,
                    nameserver='127.0.0.1',
                    result_logger=result_logger,
                    min_budget=min_budget,
                    max_budget=max_budget,
                    )

    res = bohb.run(n_iterations=n_iterations,
                   min_n_workers=params['n_workers']
                   )

    # store results
    # with open(os.path.join(shared_directory, 'results.pkl'), 'wb') as fh:
    #     pickle.dump(res, fh)

    # shutdown
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    #
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()
    all_res = res.get_all_runs()
    for i_res in range(len(all_res)):
        if all_res[i_res]['config_id'] == incumbent:
            break

    print('Best found configuration:', id2config[incumbent]['config'])
    print('A total of %i unique configurations where sampled.' %
          len(id2config.keys()))
    print('A total of %i runs where executed.' % len(res.get_all_runs()))
    print('Total budget corresponds to %.1f full function evaluations.' %
          (sum([r.budget for r in res.get_all_runs()])/max_budget))
    return id2config[incumbent]['config'], all_res[i_res]['loss']
