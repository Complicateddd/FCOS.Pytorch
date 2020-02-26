'''
@Author: xxxmy
@Github: github.com/VectXmy
@Date: 2019-10-06
@Email: xxxmy@foxmail.com
'''
##添加注释
import torch
import torch.nn as nn
from .config import DefaultConfig




def coords_fmap2orig(feature,stride):
    '''
    transfor one fmap coords to orig coords
    Args
    featurn [batch_size,h,w,c]
    stride int
    Returns 
    coords [n,2]
    '''
    h,w=feature.shape[1:3]
    shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32)
    shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = torch.reshape(shift_x, [-1])
    shift_y = torch.reshape(shift_y, [-1])
    coords = torch.stack([shift_x, shift_y], -1) + stride // 2
    return coords

class GenTargets(nn.Module):
    def __init__(self,strides,limit_range):
        super().__init__()
        self.strides=strides
        self.limit_range=limit_range
        assert len(strides)==len(limit_range)

    def forward(self,inputs):
        '''
        inputs  
        [0]list [cls_logits,cnt_logits,reg_preds]  
        cls_logits  list contains five [batch_size,class_num,h,w]  
        cnt_logits  list contains five [batch_size,1,h,w]  
        reg_preds   list contains five [batch_size,4,h,w]  
        [1]gt_boxes [batch_size,m,4]  FloatTensor  
        [2]classes [batch_size,m]  LongTensor
        Returns
        cls_targets:[batch_size,sum(_h*_w),1]
        cnt_targets:[batch_size,sum(_h*_w),1]
        reg_targets:[batch_size,sum(_h*_w),4]
        '''
        cls_logits,cnt_logits,reg_preds=inputs[0]
        gt_boxes=inputs[1]
        classes=inputs[2]
        cls_targets_all_level=[]
        cnt_targets_all_level=[]
        reg_targets_all_level=[]
        assert len(self.strides)==len(cls_logits)
        for level in range(len(cls_logits)):
            level_out=[cls_logits[level],cnt_logits[level],reg_preds[level]]
            level_targets=self._gen_level_targets(level_out,gt_boxes,classes,self.strides[level],
                                                    self.limit_range[level])
            cls_targets_all_level.append(level_targets[0])
            cnt_targets_all_level.append(level_targets[1])
            reg_targets_all_level.append(level_targets[2])
            
        return torch.cat(cls_targets_all_level,dim=1),torch.cat(cnt_targets_all_level,dim=1),torch.cat(reg_targets_all_level,dim=1)

    def _gen_level_targets(self,out,gt_boxes,classes,stride,limit_range,sample_radiu_ratio=1.5):
        '''
        Args  
        out list contains [[batch_size,class_num,h,w],[batch_size,1,h,w],[batch_size,4,h,w]]  即每一个level的list
        gt_boxes                       [batch_size,m,4]  
        classes                        [batch_size,m]  
        stride int                     level下采样的倍数
        limit_range list [min,max]     每层level 对应的limit的range
        Returns  
        cls_targets,cnt_targets,reg_targets
        '''
       
        cls_logits,cnt_logits,reg_preds=out
        batch_size=cls_logits.shape[0]
        class_num=cls_logits.shape[1]
        m=gt_boxes.shape[1]
        
        #》---------------------------------------先获得本层level下h和w对应原图的coords坐标值
        cls_logits=cls_logits.permute(0,2,3,1) #[batch_size,h,w,class_num]  
        coords=coords_fmap2orig(cls_logits,stride).to(device=gt_boxes.device)#[h*w,2]
        
        #》---------------------------------------reshape：
        cls_logits=cls_logits.reshape((batch_size,-1,class_num))#[batch_size,h*w,class_num]  
        cnt_logits=cnt_logits.permute(0,2,3,1)
        cnt_logits=cnt_logits.reshape((batch_size,-1,1))
        reg_preds=reg_preds.permute(0,2,3,1)
        reg_preds=reg_preds.reshape((batch_size,-1,4))
        
        #》--------------------------------------计算长乘宽
        h_mul_w=cls_logits.shape[1]
        
        #》--------------------------------------计算l*、t*、r*、b* ||  每一个坐标位置ij对应m个真实框的目标 i*j*m*4
        x=coords[:,0]
        y=coords[:,1]
        l_off=x[None,:,None]-gt_boxes[...,0][:,None,:]#[1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
        t_off=y[None,:,None]-gt_boxes[...,1][:,None,:]
        r_off=gt_boxes[...,2][:,None,:]-x[None,:,None]
        b_off=gt_boxes[...,3][:,None,:]-y[None,:,None]
        ltrb_off=torch.stack([l_off,t_off,r_off,b_off],dim=-1)#[batch_size,h*w,m,4]
        
        #》--------------------------------------计算每一个坐标位置ij对应m个真实框回归的出来的面积大小，用于筛选
        areas=(ltrb_off[...,0]+ltrb_off[...,2])*(ltrb_off[...,1]+ltrb_off[...,3])#[batch_size,h*w,m]
        
        #》--------------------------------------取[batch_size,h*w,m,4]四个坐标值的最大最小值，用于筛选positive
        off_min=torch.min(ltrb_off,dim=-1)[0]#[batch_size,h*w,m]
        off_max=torch.max(ltrb_off,dim=-1)[0]#[batch_size,h*w,m]
        
        #》----(mask1)--------根据最小值筛选positive：最小值大于0、即四个偏移值都大于0  即第i，j个位置必须在gt框内才算positive
        mask_in_gtboxes=off_min>0     #[batch_size,h*w,m]
        
        #》-----(mask2)-----------根据最大值是否在limit范围内筛选positive  即第框内的第i，j个位置应该离框四周距离在范围内才算positive
        mask_in_level=(off_max>limit_range[0])&(off_max<=limit_range[1])   #[batch_size,h*w,m]
        
        #》----------------------计算radiu =stride*1.5
        radiu=stride*sample_radiu_ratio
        
        #》---------------------计算框中心和第i，j位置的坐标值的偏差，正负各算一次，取最大值
        gt_center_x=(gt_boxes[...,0]+gt_boxes[...,2])/2
        gt_center_y=(gt_boxes[...,1]+gt_boxes[...,3])/2
        c_l_off=x[None,:,None]-gt_center_x[:,None,:]#[1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
        c_t_off=y[None,:,None]-gt_center_y[:,None,:]
        c_r_off=gt_center_x[:,None,:]-x[None,:,None]
        c_b_off=gt_center_y[:,None,:]-y[None,:,None]
        c_ltrb_off=torch.stack([c_l_off,c_t_off,c_r_off,c_b_off],dim=-1)#[batch_size,h*w,m,4]
        
        #》------(mask3)--------取最大值小于radiu的positive
        c_off_max=torch.max(c_ltrb_off,dim=-1)[0]
        mask_center=c_off_max<radiu
        
        #》---------------------positive: mask_in_gtboxes    &    mask_in_level    &  mask_center
        mask_pos=mask_in_gtboxes&mask_in_level&mask_center#[batch_size,h*w,m]
        
        
        #======================到此处positive筛选完成，但正样本位置i，j中#[batch_size,h*w,m]中有可能m中有两个以上为1 即一个i，j对应两个以上的框
        #======================所以筛选取其有最小面积的框作为最后回归的框
        
        #》---------把之前非正样本的area——mask置为无穷大
        areas[~mask_pos]=99999999
        
        #》---------选择具有最小area的框作为i，j回归的目标 #[batch_size,h*w] 此时tensor为h，w对应的回归框的索引
        areas_min_ind=torch.min(areas,dim=-1)[1]#[batch_size,h*w]
        
        #》------------根据areas_min_ind生成reg_targets #[batch_size,h*w,4]
        
        #改动一些bug：源代码
        #reg_targets=ltrb_off[torch.zeros_like(areas,dtype=torch.bool).scatter_(-1,areas_min_ind.unsqueeze(dim=-1),1)]#[batch_size*h*w,4]
        
        ###modify start
        tmp=torch.zeros_like(areas,dtype=torch.long).scatter_(-1,areas_min_ind.unsqueeze(dim=-1),1)
        tmp=tmp>0
        reg_targets=ltrb_off[tmp]
        ###modify  end
        reg_targets=torch.reshape(reg_targets,(batch_size,-1,4))#[batch_size,h*w,4]
        
        #》-----------根据areas_min_ind生成cls_targets #[batch_size,h*w,4]
        classes=torch.broadcast_tensors(classes[:,None,:],areas.long())[0]        #广播[batch_size,m] ->  [batch_size,h*w,m]
        
        ###改动一些bug：源代码
        #cls_targets=classes[torch.zeros_like(areas,dtype=torch.bool).scatter_(-1,areas_min_ind.unsqueeze(dim=-1),1)]
        ##modify
        cls_targets=classes[tmp]
        
        cls_targets=torch.reshape(cls_targets,(batch_size,-1,1))#[batch_size,h*w,1]
        
        #》---------------根据reg_targets计算cnt_targets
        left_right_min = torch.min(reg_targets[..., 0], reg_targets[..., 2])#[batch_size,h*w]
        left_right_max = torch.max(reg_targets[..., 0], reg_targets[..., 2])
        top_bottom_min = torch.min(reg_targets[..., 1], reg_targets[..., 3])
        top_bottom_max = torch.max(reg_targets[..., 1], reg_targets[..., 3])
        cnt_targets=((left_right_min*top_bottom_min)/(left_right_max*top_bottom_max+1e-10)).sqrt().unsqueeze(dim=-1)#[batch_size,h*w,1]
        
        
        assert reg_targets.shape==(batch_size,h_mul_w,4)
        assert cls_targets.shape==(batch_size,h_mul_w,1)
        assert cnt_targets.shape==(batch_size,h_mul_w,1)

        #process neg coords
        #》------mask_pos_2即为被选中正样本的ij位置
        mask_pos_2=mask_pos.long().sum(dim=-1)#[batch_size,h*w]
        # num_pos=mask_pos_2.sum(dim=-1)
        # assert num_pos.shape==(batch_size,)
        mask_pos_2=mask_pos_2>=1
        assert mask_pos_2.shape==(batch_size,h_mul_w)
        
        #》----------非正样本位置的分类值置为0
        cls_targets[~mask_pos_2]=0#[batch_size,h*w,1]
        
        #》----------非正样本位置的cnt和回归值置为-1
        cnt_targets[~mask_pos_2]=-1
        reg_targets[~mask_pos_2]=-1
        
        return cls_targets,cnt_targets,reg_targets
        


def compute_cls_loss(preds,targets,mask):
    '''
    Args  
    preds: list contains five level pred [batch_size,class_num,_h,_w]   5个level分开装
    targets: [batch_size,sum(_h*_w),1]  每张图片5个level的合并
    mask: [batch_size,sum(_h*_w)]       每张图片5个level的合并
    '''
    #取batchsize
    batch_size=targets.shape[0]
    
    preds_reshape=[]
    
    #取class个数
    class_num=preds[0].shape[1]
    
    #mask增加一维
    mask=mask.unsqueeze(dim=-1)
    # mask=targets>-1#[batch_size,sum(_h*_w),1]
    
    #计算每张图片5个level正样本的总数#[batch_size,]
    num_pos=torch.sum(mask,dim=[1,2]).clamp_(min=1).float()#[batch_size,]
    
    #把pred也reshape成targets的数据结构#[batch_size,sum(_h*_w),class_num]
    for pred in preds:
        pred=pred.permute(0,2,3,1)
        pred=torch.reshape(pred,[batch_size,-1,class_num])
        preds_reshape.append(pred)
    preds=torch.cat(preds_reshape,dim=1)#[batch_size,sum(_h*_w),class_num]
    
    
    assert preds.shape[:2]==targets.shape[:2]
    
    
    loss=[]
    
    #---------遍历所有图片 计算focal loss【sum,[0,0,....,1,0,0,...]】focal loss 把每个预测当做二值预测 0代表的有无，而不是背景
    for batch_index in range(batch_size):
        pred_pos=preds[batch_index]#[sum(_h*_w),class_num]
        target_pos=targets[batch_index]#[sum(_h*_w),1]
        target_pos=(torch.arange(1,class_num+1,device=target_pos.device)[None,:]==target_pos).float()#sparse-->onehot
        loss.append(focal_loss_from_logits(pred_pos,target_pos).view(1))
    return torch.cat(loss,dim=0)/num_pos#[batch_size,]

###---------------注意此处的cnt和reg_loss只针对正样本计算，即num_positive  而class_loss是整个hw每个位置ij的
def compute_cnt_loss(preds,targets,mask):
    '''
    Args  
    preds: list contains five level pred [batch_size,1,_h,_w]
    targets: [batch_size,sum(_h*_w),1]
    mask: [batch_size,sum(_h*_w)]
    '''
    batch_size=targets.shape[0]
    c=targets.shape[-1]
    preds_reshape=[]
    mask=mask.unsqueeze(dim=-1)
    # mask=targets>-1#[batch_size,sum(_h*_w),1]
    num_pos=torch.sum(mask,dim=[1,2]).clamp_(min=1).float()#[batch_size,]
    for pred in preds:
        pred=pred.permute(0,2,3,1)
        pred=torch.reshape(pred,[batch_size,-1,c])
        preds_reshape.append(pred)
    preds=torch.cat(preds_reshape,dim=1)
    assert preds.shape==targets.shape#[batch_size,sum(_h*_w),1]
    loss=[]
    for batch_index in range(batch_size):
        pred_pos=preds[batch_index][mask[batch_index]]#[num_pos_b,]
        target_pos=targets[batch_index][mask[batch_index]]#[num_pos_b,]
        assert len(pred_pos.shape)==1
        loss.append(nn.functional.binary_cross_entropy_with_logits(input=pred_pos,target=target_pos,reduction='sum').view(1))
        ##-(ylog(simoid(x))+(1-y)log(1-sigmoid(x)))
    return torch.cat(loss,dim=0)/num_pos#[batch_size,]

def compute_reg_loss(preds,targets,mask,mode='giou'):
    '''
    Args  
    preds: list contains five level pred [batch_size,4,_h,_w]
    targets: [batch_size,sum(_h*_w),4]
    mask: [batch_size,sum(_h*_w)]
    '''
    batch_size=targets.shape[0]
    c=targets.shape[-1]
    preds_reshape=[]
    # mask=targets>-1#[batch_size,sum(_h*_w),4]
    num_pos=torch.sum(mask,dim=1).clamp_(min=1).float()#[batch_size,]
    for pred in preds:
        pred=pred.permute(0,2,3,1)
        pred=torch.reshape(pred,[batch_size,-1,c])
        preds_reshape.append(pred)
    preds=torch.cat(preds_reshape,dim=1)
    assert preds.shape==targets.shape#[batch_size,sum(_h*_w),4]
    loss=[]
    for batch_index in range(batch_size):
        pred_pos=preds[batch_index][mask[batch_index]]#[num_pos_b,4]
        target_pos=targets[batch_index][mask[batch_index]]#[num_pos_b,4]
        assert len(pred_pos.shape)==2
        if mode=='iou':
            loss.append(iou_loss(pred_pos,target_pos).view(1))
        elif mode=='giou':
            loss.append(giou_loss(pred_pos,target_pos).view(1))
        else:
            raise NotImplementedError("reg loss only implemented ['iou','giou']")
    return torch.cat(loss,dim=0)/num_pos#[batch_size,]

def iou_loss(preds,targets):
    '''
    Args:
    preds: [n,4] ltrb
    targets: [n,4]
    '''
    lt=torch.min(preds[:,:2],targets[:,:2])
    rb=torch.min(preds[:,2:],targets[:,2:])
    wh=(rb+lt).clamp(min=0)
    overlap=wh[:,0]*wh[:,1]#[n]
    area1=(preds[:,2]+preds[:,0])*(preds[:,3]+preds[:,1])
    area2=(targets[:,2]+targets[:,0])*(targets[:,3]+targets[:,1])
    iou=overlap/(area1+area2-overlap)
    loss=-iou.clamp(min=1e-6).log()
    return loss.sum()

def giou_loss(preds,targets):
    '''
    Args:
    preds: [n,4] ltrb
    targets: [n,4]
    '''
    lt_min=torch.min(preds[:,:2],targets[:,:2])
    rb_min=torch.min(preds[:,2:],targets[:,2:])
    wh_min=(rb_min+lt_min).clamp(min=0)
    overlap=wh_min[:,0]*wh_min[:,1]#[n]
    area1=(preds[:,2]+preds[:,0])*(preds[:,3]+preds[:,1])
    area2=(targets[:,2]+targets[:,0])*(targets[:,3]+targets[:,1])
    union=(area1+area2-overlap)
    iou=overlap/union

    lt_max=torch.max(preds[:,:2],targets[:,:2])
    rb_max=torch.max(preds[:,2:],targets[:,2:])
    wh_max=(rb_max+lt_max).clamp(0)
    G_area=wh_max[:,0]*wh_max[:,1]#[n]

    giou=iou-(G_area-union)/G_area.clamp(1e-10)
    loss=1.-giou
    return loss.sum()

def focal_loss_from_logits(preds,targets,gamma=2.0,alpha=0.25):
    '''
    Args:
    preds: [n,class_num] 
    targets: [n,class_num]
    '''
    preds=preds.sigmoid()
    pt=preds*targets+(1.0-preds)*(1.0-targets)
    w=alpha*targets+(1.0-alpha)*(1.0-targets)
    loss=-w*torch.pow((1.0-pt),gamma)*pt.log()
    return loss.sum()




class LOSS(nn.Module):
    def __init__(self,config=None):
        super().__init__()
        if config is None:
            self.config=DefaultConfig
        else:
            self.config=config
    def forward(self,inputs):
        '''
        inputs list
        [0]preds:  ....
        [1]targets : list contains three elements [[batch_size,sum(_h*_w),1],[batch_size,sum(_h*_w),1],[batch_size,sum(_h*_w),4]]
        '''
        preds,targets=inputs
        cls_logits,cnt_logits,reg_preds=preds
        cls_targets,cnt_targets,reg_targets=targets
        mask_pos=(cnt_targets>-1).squeeze(dim=-1)# [batch_size,sum(_h*_w)]
        cls_loss=compute_cls_loss(cls_logits,cls_targets,mask_pos).mean()#[]
        cnt_loss=compute_cnt_loss(cnt_logits,cnt_targets,mask_pos).mean()
        reg_loss=compute_reg_loss(reg_preds,reg_targets,mask_pos).mean()
        if self.config.add_centerness:
            total_loss=cls_loss+cnt_loss+reg_loss
            return cls_loss,cnt_loss,reg_loss,total_loss
        else:
            total_loss=cls_loss+reg_loss+cnt_loss*0.0
            return cls_loss,cnt_loss,reg_loss,total_loss





if __name__=="__main__":
    loss=compute_cnt_loss([torch.ones([2,1,4,4])]*5,torch.ones([2,80,1]),torch.ones([2,80],dtype=torch.bool))
    print(loss)




        


        































