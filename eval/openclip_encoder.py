#!/usr/bin/env python
import torch
import torchvision
import open_clip


class OpenCLIPNetwork:
    def __init__(self, device):
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        self.clip_model_type = "ViT-B-16"
        self.clip_model_pretrained = 'laion2b_s34b_b88k'
        self.clip_n_dims = 512
        model, _, _ = open_clip.create_model_and_transforms(
            self.clip_model_type,
            pretrained=self.clip_model_pretrained,
            precision="fp16",
        )
        model.eval()
        
        self.tokenizer = open_clip.get_tokenizer(self.clip_model_type)
        self.model = model.to(device)
        self.positives = (" ",)
        # langsplat's
        #self.negatives = ("object", "things", "stuff", "texture")
        # ours
        self.negatives = ("object", "things", "sky", "background", "building", "scene")

        # "materials" 潜在的提升点的neg prompts,


        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to(device)
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to(device)
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        # embed: 32768x512
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)
        output = torch.mm(embed, p.T)
        positive_vals = output[..., positive_id : positive_id + 1]
        negative_vals = output[..., len(self.positives) :]
        repeated_pos = positive_vals.repeat(1, len(self.negatives))

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)
        softmax = torch.softmax(10 * sims, dim=-1)
        best_id = softmax[..., 0].argmin(dim=1)
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[
            :, 0, :
        ]
    @torch.no_grad()
    def get_relevancy_3d(self, embed: torch.Tensor) -> torch.Tensor:
        # embed: 32768x512
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)
        output = torch.mm(embed, p.T)
        positive_vals = output[..., 0][...,None]
        negative_vals = output[..., len(self.positives) :]
        repeated_pos = positive_vals.repeat(1,len(self.negatives))

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)
        softmax = torch.softmax(10 * sims, dim=-1)
        best_id = softmax[..., 0].argmin(dim=1)
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[
            :, 0, :
        ]
    def encode_image(self, input, mask=None):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input, mask=mask)

    def encode_text(self, text_list, device):
        text = self.tokenizer(text_list).to(device)
        return self.model.encode_text(text)
    
    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat(
                [self.tokenizer(phrase) for phrase in self.positives]
                ).to(self.neg_embeds.device)
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def set_negtives(self, text_list):
        self.negatives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat(
                [self.tokenizer(phrase) for phrase in self.negatives]
                ).to(self.pos_embeds.device)
            self.neg_embeds = self.model.encode_text(tok_phrases)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

    def set_semantics(self, text_list):
        self.semantic_labels = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.semantic_labels]).to("cuda")
            self.semantic_embeds = self.model.encode_text(tok_phrases)
        self.semantic_embeds /= self.semantic_embeds.norm(dim=-1, keepdim=True)
    
    def get_semantic_map(self, sem_map: torch.Tensor) -> torch.Tensor:
        # embed: 3xhxwx512
        n_levels, h, w, c = sem_map.shape
        pos_num = self.semantic_embeds.shape[0]
        phrases_embeds = torch.cat([self.semantic_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(sem_map.dtype)
        sem_pred = torch.zeros(n_levels, h, w)
        for i in range(n_levels):
            output = torch.mm(sem_map[i].view(-1, c), p.T)
            softmax = torch.softmax(10 * output, dim=-1)
            sem_pred[i] = torch.argmax(softmax, dim=-1).view(h, w)
            sem_pred[i][sem_pred[i] >= pos_num] = -1
        return sem_pred.long()

    def get_max_across(self, sem_map):
        n_phrases = len(self.positives)
        n_phrases_sims = [None for _ in range(n_phrases)]
        
        n_levels, h, w, _ = sem_map.shape
        clip_output = sem_map.permute(1, 2, 0, 3).flatten(0, 1)

        n_levels_sims = [None for _ in range(n_levels)]
        for i in range(n_levels):
            for j in range(n_phrases):
                probs = self.get_relevancy(clip_output[..., i, :], j)
                pos_prob = probs[..., 0:1]
                n_phrases_sims[j] = pos_prob
            n_levels_sims[i] = torch.stack(n_phrases_sims)
        
        relev_map = torch.stack(n_levels_sims).view(n_levels, n_phrases, h, w)
        return relev_map

    def get_max_across_post(self, sem_map_list,is_sky_filter = False):
        """
        Given a list of the tensors,
        cauculate the valid_map for each lvl; each text_prompt; each Virtual Apeearance Renderings

        Args:
            sem_map_list (Tensor[n_var;n_lvl;w,h;512]) : The input clip feature
            is_sky_filter (Bool) : If True, the "sky" will also be cauculated as the positive prompts, however, (we need to del all neg_prompts when cauc sky)
        Returns:
            relev_map (Tensor[n_var,n_lvl,n_text_prompts,W,H]: The corresponding relevancy map.

        Example:
            INPUT: sem_map_list[4,3,528,390,512]

            4 appearances( Origin + Novel Appearance 1 + Novel Appearance 2 + Novel Appearance 3)
            3 lvls from the SAM (low, medium, high)
            2 text_prompts
            [390, 512] Image Weigh and Heigh
            OUTPUT： -> relev_map [4,3,2,390,512]

        """
        # Input sem_map:[4,3,528,390,512]
        if is_sky_filter:
            n_phrases = len(self.positives) - 1 # except "sky"
        else:
            n_phrases = len(self.positives)


        n_nar, n_levels, h, w, _ = sem_map_list.shape # Nar: Novel Appearance Rendering (commonly: 4:[default;NAR1;NAR2;NAR3])
        clip_output = sem_map_list.permute(2, 3, 0, 1, 4).flatten(0,1) # [4,3,W,H,512]  sem_map.permute(2, 3, 0, 1, 4):[W,H,3,4,512]->flatten(0,1):[W*H,3,4,512]
        clip_output = clip_output.view(w * h, n_nar * n_levels, -1) # [w*h, 12, 512]
        n_levels_sims = [None for _ in range(n_nar * n_levels)] # return n_level x n _nar; 3 x 4 = 12
        n_phrases_sims = [None for _ in range(n_phrases)]  # List:[] len: len of n_text
        for i in range(n_nar * n_levels):# 3 * 4 = 12
            for j in range(n_phrases):# num_prompt
                probs = self.get_relevancy(clip_output[..., i, :], j) # clip_output[..., i, :]:i-th level [w*h,512]; j:text_prompt
                pos_prob = probs[..., 0:1] #probs[N,2];pos_prob[N,1]
                n_phrases_sims[j] = pos_prob
            n_levels_sims[i] = torch.stack(n_phrases_sims)
        # n_levels_sims List3 item:[2,N,1]
        # torch.stack(n_levels_sims):[12,2,N,1]->[12,2,h,w]
        relev_map = torch.stack(n_levels_sims).view(n_nar,n_levels, n_phrases, h, w)
        n_levels_sims = [None for _ in range(n_nar * n_levels)] # return n_level x n _nar; 3 x 4 = 12

        if is_sky_filter:

            record_neg_list = self.negatives
            record_pos_list = self.positives

            self.set_positives(["sky", "background"])
            #self.set_negtives(["Iron Cross","column","Bronze sculpture"] )
            self.set_negtives(record_pos_list)
            for i in range(n_nar * n_levels):# 3 * 4 = 12
                probs = self.get_relevancy(clip_output[..., i, :],
                                           0)  # clip_output[..., i, :]:i-th level [w*h,512]; j:text_prompt
                pos_prob = probs[..., 0:1]  # probs[N,2];pos_prob[N,1]
                n_levels_sims[i] = pos_prob
            relev_map_sky = torch.stack(n_levels_sims).view(n_nar, n_levels, 1, h, w)
            # attention to roll back
            self.set_positives(record_pos_list)
            self.set_negtives(record_neg_list)
            relev_map = torch.cat([relev_map,relev_map_sky],dim = 2)
        return relev_map

    def get_max_across_3d_post(self, clip_output, is_sky_filter=False):#[n,3,4,512]
        # design for only one key
        """
        Given a list of the tensors,
        cauculate the valid_map for each lvl; each text_prompt; each Virtual Apeearance Renderings

        Args:
            sem_map_list (Tensor[n_var;n_lvl;w,h;512]) : The input 3D clip feature [N,204]
            is_sky_filter (Bool) : If True, the "sky" will also be cauculated as the positive prompts, however, (we need to del all neg_prompts when cauc sky)
        Returns:
            relev_map (Tensor[n_var,n_lvl,n_text_prompts,W,H]: The corresponding relevancy map.

        Example:
            INPUT: sem_map_list[4,3,528,390,512]  ->[4,3,N_ptr,512]

            4 appearances( Origin + Novel Appearance 1 + Novel Appearance 2 + Novel Appearance 3)
            3 lvls from the SAM (low, medium, high)

            [390, 512] Image Weigh and Heigh
            OUTPUT： -> relev_map [4,3,2,390,512]

        """
        # Input sem_map:[4,3,528,390,512]
        clip_output = clip_output.view(-1,12,512)
        if is_sky_filter:
            n_phrases = len(self.positives) - 1  # except "sky"
        else:
            n_phrases = len(self.positives)

        n_levels_sims = [None for _ in range(12)]
        for i in range(12):  # 3 * 4 = 12
                probs = self.get_relevancy_3d(clip_output[:,i,:])  # clip_output[..., i, :]:i-th level [w*h,512]; j:text_prompt
                pos_prob = probs[..., 0:1]  # probs[N,2];pos_prob[N,1]
                n_levels_sims[i] = pos_prob
        # n_levels_sims List3 item:[2,N,1]
        # torch.stack(n_levels_sims):[12,2,N,1]->[12,2,h,w]
        relev_map = torch.stack(n_levels_sims).view(12, -1)
        return relev_map



