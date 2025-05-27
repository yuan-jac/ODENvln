import collections

import torch.nn as nn

from .vlnbert_init import get_vlnbert_models


class VLNBert(nn.Module):
    def __init__(self, args):
        super().__init__()
        print('\nInitalizing the VLN-BERT model ...')
        self.args = args

        self.vln_bert = get_vlnbert_models(args, config=None)  # initialize the VLN-BERT
        self.drop_env = nn.Dropout(p=args.feat_dropout)
        self.drop_prob = args.feat_dropout

    def shared_dropout(self, feat_img, feat_text):
        if not self.training or self.drop_prob == 0.:
            return feat_img, feat_text

        # 生成同一掩码
        mask = feat_img.new_empty(feat_img.shape).bernoulli_(1 - self.drop_prob)
        mask = mask / (1 - self.drop_prob)

        feat_img = feat_img * mask
        feat_text = feat_text * mask

        return feat_img, feat_text

    def forward(self, mode, batch):

        batch = collections.defaultdict(lambda: None, batch)

        if mode == 'language':
            txt_embeds = self.vln_bert(mode, batch)
            return txt_embeds

        elif mode == 'panorama':
            batch['view_img_fts'], batch['view_text_fts'] = self.shared_dropout(
                batch['view_img_fts'], batch['view_text_fts'])
            if 'obj_img_fts' in batch:
                batch['obj_img_fts'] = self.drop_env(batch['obj_img_fts'])
            pano_embeds, pano_masks = self.vln_bert(mode, batch)
            return pano_embeds, pano_masks

        elif mode == 'navigation':
            outs = self.vln_bert(mode, batch)
            return outs

        else:
            raise NotImplementedError('wrong mode: %s'%mode)


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()
