import timm
import torch
from timm.models.vision_transformer import VisionTransformer

# Wrappermodel to execute timm models
class wrappermodel(torch.nn.Module):
    def __init__(self,num_classes,weight_path=None):
        super().__init__()
        self.basemodel = VisionTransformer(img_size=32,patch_size=4,num_classes=num_classes,embed_dim=192, depth=12, num_heads=3)
        if weight_path is not None:
            sd = torch.load(weight_path,map_location='cpu')
            self.basemodel.load_state_dict(sd)

    def forward(self,x,return_feature=False,return_feature_list=False):
        if return_feature:
            out = self.basemodel.forward_features(x)
            features = self.basemodel.forward_head(out,pre_logits=True)
            logits = self.basemodel.forward_head(out,pre_logits=False)
            return logits, features
        else: 
            logits = self.basemodel.forward(x)
            return logits

    def get_fc(self):
        fc = self.basemodel.get_classifier()
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.basemodel.get_classifier()
    
    def forward_threshold(self, x, threshold):
        out = self.basemodel.forward_features(x)
        out = out.clip(max=threshold)
        logits = self.basemodel.forward_head(out,pre_logits=False)
        return logits
    
    

    