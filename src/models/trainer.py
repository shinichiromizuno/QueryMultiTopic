import os
import glob

import numpy as np
import torch
from tensorboardX import SummaryWriter

import distributed
# import onmt
from models.reporter import ReportMgr
from models.stats import Statistics
from models import data_loader
from models.data_loader import load_test_dataset
from others.logging import logger
from others.utils import test_rouge, rouge_results_to_str
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params


def build_trainer(args, device_id, model,
                  optim):
    """
    Simplify `Trainer` creation based on user `opt`s*
    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """
    device = "cpu" if args.visible_gpus == '-1' else "cuda"


    grad_accum_count = args.accum_count
    n_gpu = args.world_size

    if device_id >= 0:
        gpu_rank = int(args.gpu_ranks[device_id])
    else:
        gpu_rank = 0
        n_gpu = 0

    print('gpu_rank %d' % gpu_rank)

    tensorboard_log_dir = args.model_path

    writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

    report_manager = ReportMgr(args.report_every, start_time=-1, tensorboard_writer=writer)

    trainer = Trainer(args, model, optim, grad_accum_count, n_gpu, gpu_rank, report_manager)

    # print(tr)
    if (model):
        n_params = _tally_parameters(model)
        logger.info('* number of parameters: %d' % n_params)

    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self,  args, model,  optim,
                  grad_accum_count=1, n_gpu=1, gpu_rank=1,
                  report_manager=None):
        # Basic attributes.
        self.args = args
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.model = model
        self.optim = optim
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager

        self.loss = torch.nn.BCELoss(reduction='none')
        assert grad_accum_count > 0
        # Set model in training mode.
        if (model):
            self.model.train()

    def train(self, train_iter_fct, train_steps, valid_iter_fct=None, valid_steps=-1):
        """
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter_fct`

        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter_fct(function): same as train_iter_fct, for valid data
            train_steps(int):
            valid_steps(int):
            save_checkpoint_steps(int):

        Return:
            None
        """
        logger.info('Start training...!!!')

        # step =  self.optim._step + 1
        step =  self.optim._step + 1
        true_batchs = []
        accum = 0
        normalization = 0
        train_iter = train_iter_fct()

        total_stats = Statistics()
        report_stats = Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        print(f'Initial Step:{step}, Train Steps:{train_steps}')
        while step <= train_steps:

            reduce_counter = 0
            for i, batch in enumerate(train_iter):
                if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):
                    true_batchs.append(batch)
                    normalization += batch.batch_size
                    accum += 1
                    if accum == self.grad_accum_count:
                        reduce_counter += 1
                        if self.n_gpu > 1:
                            normalization = sum(distributed
                                                .all_gather_list
                                                (normalization))

                        self._gradient_accumulation(
                            true_batchs, normalization, total_stats,
                            report_stats)

                        report_stats = self._maybe_report_training(
                            step, train_steps,
                            self.optim.learning_rate,
                            report_stats)

                        true_batchs = []
                        accum = 0
                        normalization = 0
                        if (step % self.save_checkpoint_steps == 0 and self.gpu_rank == 0):
                            self._save(step)

                        step += 1
                        if step > train_steps:
                            break
            train_iter = train_iter_fct()

        return total_stats

    def validate(self, valid_iter, step=0):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()
        stats = Statistics()

        with torch.no_grad():
            for batch in valid_iter:

                src = batch.src
                labels = batch.labels
                segs = batch.segs
                clss = batch.clss
                mask = batch.mask
                mask_cls = batch.mask_cls

                sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)


                loss = self.loss(sent_scores, labels.float())
                loss = (loss * mask.float()).sum()
                batch_stats = Statistics(float(loss.cpu().data.numpy()), len(labels))
                stats.update(batch_stats)
            self._report_step(0, step, valid_stats=stats)
            return stats

    # def test(self, test_iter, step, threshold=0.5, cal_lead=False, cal_oracle=False):
    #     """ Validate model.
    #         valid_iter: validate data iterator
    #     Returns:
    #         :obj:`nmt.Statistics`: validation loss statistics
    #     """
    #     # Set model in validating mode.
    #     def _get_ngrams(n, text):
    #         ngram_set = set()
    #         text_length = len(text)
    #         max_index_ngram_start = text_length - n
    #         for i in range(max_index_ngram_start + 1):
    #             ngram_set.add(tuple(text[i:i + n]))
    #         return ngram_set

    #     def _block_tri(c, p):
    #         tri_c = _get_ngrams(3, c.split())
    #         for s in p:
    #             tri_s = _get_ngrams(3, s.split())
    #             if len(tri_c.intersection(tri_s))>0:
    #                 return True
    #         return False
        
    #     if (not cal_lead and not cal_oracle):
    #         self.model.eval()
    #     stats = Statistics()

    #     can_path = '%s_step%d.candidate'%(self.args.result_path,step)
    #     gold_path = '%s_step%d.gold' % (self.args.result_path, step)
    #     with open(can_path, 'w') as save_cand:
    #         with open(gold_path, 'w') as save_gold:
    #             with torch.no_grad():
    #                 # pred_list = []
    #                 # labels_list = []
    #                 # cand_list = []
    #                 # gold_list = []

    #                 optim_dicts = {}
    #                 for i in range(1, 18):
    #                     optim_dicts[i] = {}

    #                 for batch in test_iter:
    #                     src = batch.src
    #                     labels = batch.labels
    #                     segs = batch.segs
    #                     clss = batch.clss
    #                     mask = batch.mask
    #                     mask_cls = batch.mask_cls

    #                     qry = batch.qry
    #                     stpos = batch.stpos
    #                     enpos = batch.enpos

    #                     # gold = []
    #                     # pred = []

    #                     # if (cal_lead):
    #                     #     selected_ids = [list(range(batch.clss.size(1)))] * batch.batch_size
    #                     # elif (cal_oracle):
    #                     #     selected_ids = [[j for j in range(batch.clss.size(1)) if labels[i][j] == 1] for i in
    #                     #                     range(batch.batch_size)]
    #                     # else:
    #                     #     sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)

    #                     #     loss = self.loss(sent_scores, labels.float())
    #                     #     loss = (loss * mask.float()).sum()
    #                     #     batch_stats = Statistics(float(loss.cpu().data.numpy()), len(labels))
    #                     #     stats.update(batch_stats)

    #                     #     print(f'labels:{labels}')
    #                     #     print(f'sent_scores:{sent_scores}')
    #                     #     print(f'src_str:{batch.src_str}')
    #                     #     # sent_scores = sent_scores + mask.float()
    #                     #     sent_scores = sent_scores.cpu().data.numpy()

    #                     sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)

    #                     loss = self.loss(sent_scores, labels.float())
    #                     loss = (loss * mask.float()).sum()
    #                     batch_stats = Statistics(float(loss.cpu().data.numpy()), len(labels))
    #                     stats.update(batch_stats)

    #                     # print(f'labels:{labels}')
    #                     # print(f'sent_scores:{sent_scores}')
    #                     # print(f'src_str:{batch.src_str}')
    #                     # sent_scores = sent_scores + mask.float()
    #                     sent_scores = sent_scores.cpu().data.numpy()

    #                         # selected_ids = np.argsort(-sent_scores, 1)
                        
    #                     # for i, idx in enumerate(selected_ids):
    #                     #     _pred = []
    #                     #     if(len(batch.src_str[i])==0):
    #                     #         continue
    #                     #     for j in selected_ids[i][:len(batch.src_str[i])]:
    #                     #         if(j>=len( batch.src_str[i])):
    #                     #             continue
    #                     #         candidate = batch.src_str[i][j].strip()
    #                     #         if(self.args.block_trigram):
    #                     #             if(not _block_tri(candidate,_pred)):
    #                     #                 _pred.append(candidate)
    #                     #         else:
    #                     #             _pred.append(candidate)

    #                     #         if ((not cal_oracle) and (not self.args.recall_eval) and len(_pred) == 3):
    #                     #             break

    #                     #     _pred = '<q>'.join(_pred)
    #                     #     if(self.args.recall_eval):
    #                     #         _pred = ' '.join(_pred.split()[:len(batch.tgt_str[i].split())])

    #                     #     pred.append(_pred)
    #                     #     gold.append(batch.tgt_str[i])

    #                     # thres = 0.5
    #                     # labels = labels.cpu().data.numpy()

    #                     # for i in batch.src_str:
    #                     #     print(len(i))

    #                     # src_str = np.array(batch.src_str)
    #                     # print(src_str.size)
    #                     pred = np.where(sent_scores < threshold, 0, 1)
    #                     # cand = []
    #                     # print(f'pred_list:{pred}')
    #                     # for i in range(len(src_str)):
    #                     #     cand.append(src_str[i][np.where(pred[i] == 1, True, False)].tolist())
    #                     # gold = []
    #                     # for i in range(len(src_str)):
    #                     #     gold.append(src_str[i][np.where(labels[i] == 1, True, False)].tolist())

    #                     # print(f'gold:{gold}')
                        
    #                     # for i in range(len(pred)):
    #                     #     pred_list += pred.tolist()[i]
    #                     # for i in range(len(labels)):
    #                     #     labels_list += labels.tolist()[i]

    #                     # print(f'cand:{cand}')
    #                     # print(f'gold:{gold}')
    #                     # print(batch.qry)
    #                     # print(batch.stpos)
    #                     # print(batch.enpos)

    #                     # for j in range(len(cand)):
    #                     #     cand_list += 'qry:' + qry[i] + ' stpos:' + str(stpos[i]) + ' enpos:' + str(enpos[i]) + '\n'
    #                     #     cand_list += ' '.join(cand[i])+'\n'
    #                     # for j in range(len(gold)):
    #                     #     gold_list += 'qry:' + qry[i] + ' stpos:' + str(stpos[i]) + ' enpos:' + str(enpos[i]) + '\n'
    #                     #     gold_list += ' '.join(gold[i])+'\n'

    #                     # Update or add pred and label to dictionary by qry and position
    #                     # for j in range(stpos[0], enpos[0]):
    #                     #     st_idx = j - stpos[i]
    #                     #     en_idx = enpos[i] - 1 - j
    #                     #     pred_eval = st_idx * en_idx
    #                     #     if optim_dicts[qry[0]].get(j):
    #                     #         if optim_dicts[qry[0]][j]['pred'][-1] < pred_eval:
    #                     #             optim_dicts[qry[0]][j]['pred'] = [pred[st_idx], pred_eval]
    #                     #     else:
    #                     #         optim_dicts[qry[0]][j] = {'pred':[pred[st_idx], pred_eval], 'labels':labels[st_idx]}    

    #                     for j in range(stpos[0], enpos[0]):
    #                         st_idx = j - stpos[i]
    #                         en_idx = enpos[i] - 1 - j
    #                         pred_eval = st_idx * en_idx
    #                         if optim_dicts[qry[0]].get(j):
    #                             if optim_dicts[qry[0]][j][-1] < pred_eval:
    #                                 optim_dicts[qry[0]][j] = [pred[st_idx], pred_eval]
    #                         else:
    #                             optim_dicts[qry[0]][j] = [pred[st_idx], pred_eval]   

    #                     # for i in range(len(gold)):
    #                     #     save_gold.write(gold[i].strip()+'\n')
    #                     # for i in range(len(pred)):
    #                     #     save_pred.write(pred[i].strip()+'\n')

    #                     # for i in range(len(gold)):
    #                     #     save_gold.write(' '.join(gold[i])+'\n')
    #                     # for i in range(len(cand)):
    #                     #     save_cand.write(' '.join(cand[i])+'\n')

    #                 # print(f'labels_list:{labels_list}')
    #                 # print(f'pred_list:{pred_list}')

    #                 # dictionaryからpred_listとlabels_listを作り直す処理
    #                 src = test_iter.src
    #                 print(f'src:{src}')
    #                 cand_list = []
    #                 gold_list = []
    #                 for i in range(1, 18):
    #                     pred_all = []
    #                     # labels_list = []
    #                     tgt = test_iter.tgts[str(i)]
    #                     print(f'tgt:{tgt}')
    #                     for j in sorted(optim_dicts[i].keys()):
    #                         # pred_list.append(optim_dicts[i][j]['pred'][0])
    #                         # labels_list.append(optim_dicts[i][j]['labels'])
    #                         pred_all.append(optim_dicts[i][j][0])

    #                     print(f'accuracy_score:{accuracy_score(tgt, pred_all)}')
    #                     print(f'confusion_matrix:{confusion_matrix(tgt, pred_all, labels=[1, 0])}')
    #                     print(f'f1_score:{f1_score(tgt, pred_all)}')

    #                     cand = []
    #                     print(f'pred_all:{pred_all}')
    #                     for i in range(len(src)):
    #                         cand.append(src[i][np.where(pred_all[i] == 1, True, False)].tolist())
    #                     cand_list.append(cand)                
    #                     gold = []
    #                     for i in range(len(src)):
    #                         gold.append(src[i][np.where(tgt[i] == 1, True, False)].tolist())
    #                     gold_list.append(gold)                
                    
    #                 save_cand.write(''.join(cand_list))
    #                 save_gold.write(''.join(gold_list))
    #     # Rouge Calculation
    #     # if(step!=-1 and self.args.report_rouge):
    #     #     rouges = test_rouge(self.args.temp_dir, can_path, gold_path)
    #     #     logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
    #     self._report_step(0, step, valid_stats=stats)

    #     return stats

    def test(self, args, step):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i:i + n]))
            return ngram_set

        def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s))>0:
                    return True
            return False
        
        # if (not cal_lead and not cal_oracle):
        #     self.model.eval()
        device = "cpu" if args.visible_gpus == '-1' else "cuda"
        stats = Statistics()

        can_path = '%s_step%d.candidate'%(self.args.result_path,step)
        gold_path = '%s_step%d.gold' % (self.args.result_path, step)
        with open(can_path, 'w') as save_cand:
            with open(gold_path, 'w') as save_gold:
                with torch.no_grad():
                    pts = sorted(glob.glob(args.bert_data_path + 'test' + '.*.pt'))
                    for pt in pts:
                        dataset = torch.load(pt)
                        ds = dataset['ds']
                        src_txt = dataset['src']
                        tgts = dataset['tgts']
                        # test_iter = data_loader.Dataloader(args, iter(dataset['ds']), args.batch_size, device,
                        #                                 shuffle=True, is_test=False)
                        logger.info('Loading %s dataset from %s, number of examples: %d' %
                                                ('test', pt, len(ds)))
                        test_iter = data_loader.Dataloader(args, load_test_dataset(ds), args.batch_size, device,
                                                        shuffle=True, is_test=True)
                        optim_dicts = {}
                        for i in tgts.keys():
                            optim_dicts[i] = {}

                        for batch in test_iter:
                            src = batch.src
                            labels = batch.labels
                            segs = batch.segs
                            clss = batch.clss
                            mask = batch.mask
                            mask_cls = batch.mask_cls

                            qry = batch.qry
                            stpos = batch.stpos
                            enpos = batch.enpos

                            sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)

                            loss = self.loss(sent_scores, labels.float())
                            loss = (loss * mask.float()).sum()
                            batch_stats = Statistics(float(loss.cpu().data.numpy()), len(labels))
                            stats.update(batch_stats)

                            sent_scores = sent_scores.cpu().data.numpy()
                            pred = np.where(sent_scores < args.threshold, 0, 1) 
                            
                            for j in range(stpos[0], enpos[0]):
                                st_idx = j - stpos[0]
                                en_idx = enpos[0] - 1 - j
                                # en_idx = enpos[0] - j
                                pred_eval = st_idx * en_idx
                                if optim_dicts[qry[0]].get(j):
                                    if optim_dicts[qry[0]][j][-1] < pred_eval:
                                        optim_dicts[qry[0]][j] = [pred[0][st_idx], pred_eval]
                                else:
                                    # print(f'j:{j}, stpos:{stpos[0]}, enpos:{enpos[0]}, st_idx:{st_idx}')
                                    # print(f'pred:{pred}')
                                    # print(f'optim_dicts:{optim_dicts}')
                                    optim_dicts[qry[0]][j] = [pred[0][st_idx], pred_eval]   

                        # dictionaryからpred_listとlabels_listを作り直す処理
                        save_cand.write(pt+'\n')
                        save_gold.write(pt+'\n')
                        # cand_list = []
                        # gold_list = []
                        for i in tgts.keys():
                            pred_all = []
                            tgt = dataset['tgts'][i]
                            # print(f'tgt:{tgt}')
                            # print(f'optim_dicts[i]:{optim_dicts[i]}')
                            for j in sorted(optim_dicts[i].keys()):
                                pred_all.append(optim_dicts[i][j][0])

                            # print(f'tgt:{tgt}, pred_all:{pred_all}')

                            # print(f'tgt:{i}')
                            # print(f'accuracy_score:{accuracy_score(tgt, pred_all)}')
                            # print(f'confusion_matrix:{confusion_matrix(tgt, pred_all, labels=[1, 0])}')
                            # print(f'f1_score:{f1_score(tgt, pred_all)}')
                            logger.info('tgt:%s', i)
                            logger.info('accuracy_score:%s', accuracy_score(tgt, pred_all))
                            logger.info('confusion_matrix:%s', confusion_matrix(tgt, pred_all, labels=[1, 0]))
                            logger.info('f1_score:%s', f1_score(tgt, pred_all))

                            cand = [i for i, j in zip(src_txt, pred_all) if j == 1]
                            # cand_list.append(cand)                
                            gold = [i for i, j in zip(src_txt, tgt) if j == 1]
                            # gold_list.append(gold)

                            save_cand.write('qry'+i+'\n'+''.join(cand)+'\n')
                            save_gold.write('qry'+i+'\n'+''.join(gold)+'\n')

                        # print(f'cand_list:{cand_list}')
                        # print(f'gold_list:{gold_list}')
                        # raise Exception('Batch stops')              
                    
                        # save_cand.write(''.join(cand_list)+'\n')
                        # save_gold.write(''.join(gold_list)+'\n')
        # Rouge Calculation
        # if(step!=-1 and self.args.report_rouge):
        #     rouges = test_rouge(self.args.temp_dir, can_path, gold_path)
        #     logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
        self._report_step(0, step, valid_stats=stats)

        return stats



    def _gradient_accumulation(self, true_batchs, normalization, total_stats,
                               report_stats):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            if self.grad_accum_count == 1:
                self.model.zero_grad()

            src = batch.src
            labels = batch.labels
            segs = batch.segs
            clss = batch.clss
            mask = batch.mask
            mask_cls = batch.mask_cls

            sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)

            loss = self.loss(sent_scores, labels.float())
            loss = (loss*mask.float()).sum()
            (loss/loss.numel()).backward()
            # loss.div(float(normalization)).backward()

            batch_stats = Statistics(float(loss.cpu().data.numpy()), normalization)


            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            # 4. Update the parameters and statistics.
            if self.grad_accum_count == 1:
                # Multi GPU gradient gather
                if self.n_gpu > 1:
                    grads = [p.grad.data for p in self.model.parameters()
                             if p.requires_grad
                             and p.grad is not None]
                    distributed.all_reduce_and_rescale_tensors(
                        grads, float(1))
                self.optim.step()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.grad_accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            self.optim.step()

    def _save(self, step):
        real_model = self.model
        # real_generator = (self.generator.module
        #                   if isinstance(self.generator, torch.nn.DataParallel)
        #                   else self.generator)

        model_state_dict = real_model.state_dict()
        # generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            # 'generator': generator_state_dict,
            'opt': self.args,
            'optim': self.optim,
        }
        checkpoint_path = os.path.join(self.args.model_path, 'model_step_%d.pt' % step)
        logger.info("Saving checkpoint %s" % checkpoint_path)
        # checkpoint_path = '%s_step_%d.pt' % (FLAGS.model_path, step)
        if (not os.path.exists(checkpoint_path)):
            torch.save(checkpoint, checkpoint_path)
            return checkpoint, checkpoint_path

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)
