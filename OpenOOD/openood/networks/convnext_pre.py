import timm
import torch

class convnext_1kpre(torch.nn.Module):
    def __init__(self,num_classes=10,weight_path=None):
        super().__init__()
        self.basemodel = timm.create_model("hf_hub:anonauthors/cifar10-timm-convnext_base.fb_in1k", pretrained=True)

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
        fc = self.basemodel.head.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.head.fc
    

    