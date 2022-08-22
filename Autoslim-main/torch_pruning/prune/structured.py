import torch #line:1
import torch .nn as nn #line:2
from copy import deepcopy #line:3
from functools import reduce #line:4
from operator import mul #line:5
__all__ =['prune_conv','prune_related_conv','prune_linear','prune_related_linear','prune_batchnorm','prune_prelu','prune_group_conv']#line:7
def prune_group_conv (OOOO0OOOOO00000OO :nn .modules .conv ._ConvNd ,OOOO00OO00O00O000 :list ,inplace :bool =True ,dry_run :bool =False ):#line:9
    ""#line:15
    if OOOO0OOOOO00000OO .groups >1 :#line:16
         assert OOOO0OOOOO00000OO .groups ==OOOO0OOOOO00000OO .in_channels and OOOO0OOOOO00000OO .groups ==OOOO0OOOOO00000OO .out_channels ,"only group conv with in_channel==groups==out_channels is supported"#line:17
    OOOO00OO00O00O000 =list (set (OOOO00OO00O00O000 ))#line:19
    OOO0OO0O0OO0O000O =len (OOOO00OO00O00O000 )*reduce (mul ,OOOO0OOOOO00000OO .weight .shape [1 :])+(len (OOOO00OO00O00O000 )if OOOO0OOOOO00000OO .bias is not None else 0 )#line:20
    if dry_run :#line:21
        return OOOO0OOOOO00000OO ,OOO0OO0O0OO0O000O #line:22
    if not inplace :#line:23
        OOOO0OOOOO00000OO =deepcopy (OOOO0OOOOO00000OO )#line:24
    OOOO0000000000OO0 =[O0O0OO0OO0O00O0O0 for O0O0OO0OO0O00O0O0 in range (OOOO0OOOOO00000OO .out_channels )if O0O0OO0OO0O00O0O0 not in OOOO00OO00O00O000 ]#line:25
    OOOO0OOOOO00000OO .out_channels =OOOO0OOOOO00000OO .out_channels -len (OOOO00OO00O00O000 )#line:26
    OOOO0OOOOO00000OO .in_channels =OOOO0OOOOO00000OO .in_channels -len (OOOO00OO00O00O000 )#line:27
    OOOO0OOOOO00000OO .groups =OOOO0OOOOO00000OO .groups -len (OOOO00OO00O00O000 )#line:28
    OOOO0OOOOO00000OO .weight =torch .nn .Parameter (OOOO0OOOOO00000OO .weight .data .clone ()[OOOO0000000000OO0 ])#line:29
    if OOOO0OOOOO00000OO .bias is not None :#line:30
        OOOO0OOOOO00000OO .bias =torch .nn .Parameter (OOOO0OOOOO00000OO .bias .data .clone ()[OOOO0000000000OO0 ])#line:31
    return OOOO0OOOOO00000OO ,OOO0OO0O0OO0O000O #line:32
def prune_conv (OOOOOOOO00000O00O :nn .modules .conv ._ConvNd ,OO00OO0OOOO0000O0 :list ,inplace :bool =True ,dry_run :bool =False ):#line:34
    ""#line:40
    OO00OO0OOOO0000O0 =list (set (OO00OO0OOOO0000O0 ))#line:41
    O00O00OO00O00OO0O =len (OO00OO0OOOO0000O0 )*reduce (mul ,OOOOOOOO00000O00O .weight .shape [1 :])+(len (OO00OO0OOOO0000O0 )if OOOOOOOO00000O00O .bias is not None else 0 )#line:42
    if dry_run :#line:43
        return OOOOOOOO00000O00O ,O00O00OO00O00OO0O #line:44
    if not inplace :#line:46
        OOOOOOOO00000O00O =deepcopy (OOOOOOOO00000O00O )#line:47
    O000O0O0OO0O0000O =[OO0OOO0OO0O0OOO0O for OO0OOO0OO0O0OOO0O in range (OOOOOOOO00000O00O .out_channels )if OO0OOO0OO0O0OOO0O not in OO00OO0OOOO0000O0 ]#line:49
    OOOOOOOO00000O00O .out_channels =OOOOOOOO00000O00O .out_channels -len (OO00OO0OOOO0000O0 )#line:50
    if isinstance (OOOOOOOO00000O00O ,(nn .ConvTranspose2d ,nn .ConvTranspose3d )):#line:51
        OOOOOOOO00000O00O .weight =torch .nn .Parameter (OOOOOOOO00000O00O .weight .data .clone ()[:,O000O0O0OO0O0000O ])#line:52
    else :#line:53
        OOOOOOOO00000O00O .weight =torch .nn .Parameter (OOOOOOOO00000O00O .weight .data .clone ()[O000O0O0OO0O0000O ])#line:54
    if OOOOOOOO00000O00O .bias is not None :#line:55
        OOOOOOOO00000O00O .bias =torch .nn .Parameter (OOOOOOOO00000O00O .bias .data .clone ()[O000O0O0OO0O0000O ])#line:56
    return OOOOOOOO00000O00O ,O00O00OO00O00OO0O #line:57
def prune_related_conv (OO000O000O0O0O0O0 :nn .modules .conv ._ConvNd ,O0O0000000O0O0OO0 :list ,inplace :bool =True ,dry_run :bool =False ):#line:59
    ""#line:65
    O0O0000000O0O0OO0 =list (set (O0O0000000O0O0OO0 ))#line:66
    O0OO00O0O0OOOO000 =len (O0O0000000O0O0OO0 )*OO000O000O0O0O0O0 .weight .shape [0 ]*reduce (mul ,OO000O000O0O0O0O0 .weight .shape [2 :])#line:67
    if dry_run :#line:68
        return OO000O000O0O0O0O0 ,O0OO00O0O0OOOO000 #line:69
    if not inplace :#line:70
        OO000O000O0O0O0O0 =deepcopy (OO000O000O0O0O0O0 )#line:71
    OO0O0OO0OOO0OO0O0 =[O0O0000OOO0OOOO0O for O0O0000OOO0OOOO0O in range (OO000O000O0O0O0O0 .in_channels )if O0O0000OOO0OOOO0O not in O0O0000000O0O0OO0 ]#line:74
    OO000O000O0O0O0O0 .in_channels =OO000O000O0O0O0O0 .in_channels -len (O0O0000000O0O0OO0 )#line:76
    if isinstance (OO000O000O0O0O0O0 ,(nn .ConvTranspose2d ,nn .ConvTranspose3d )):#line:78
        OO000O000O0O0O0O0 .weight =torch .nn .Parameter (OO000O000O0O0O0O0 .weight .data .clone ()[OO0O0OO0OOO0OO0O0 ,:])#line:79
    else :#line:80
        OO000O000O0O0O0O0 .weight =torch .nn .Parameter (OO000O000O0O0O0O0 .weight .data .clone ()[:,OO0O0OO0OOO0OO0O0 ])#line:81
    return OO000O000O0O0O0O0 ,O0OO00O0O0OOOO000 #line:83
def prune_linear (O0O0OOO00OO000OOO :nn .modules .linear .Linear ,O0O0OOOO00O0O0OOO :list ,inplace :list =True ,dry_run :list =False ):#line:85
    ""#line:91
    O0O0000OOO000OO00 =len (O0O0OOOO00O0O0OOO )*O0O0OOO00OO000OOO .weight .shape [1 ]+(len (O0O0OOOO00O0O0OOO )if O0O0OOO00OO000OOO .bias is not None else 0 )#line:92
    if dry_run :#line:93
        return O0O0OOO00OO000OOO ,O0O0000OOO000OO00 #line:94
    if not inplace :#line:96
        O0O0OOO00OO000OOO =deepcopy (O0O0OOO00OO000OOO )#line:97
    OOOO00O0OOOOO00OO =[O0O0O000OO0000O0O for O0O0O000OO0000O0O in range (O0O0OOO00OO000OOO .out_features )if O0O0O000OO0000O0O not in O0O0OOOO00O0O0OOO ]#line:98
    O0O0OOO00OO000OOO .out_features =O0O0OOO00OO000OOO .out_features -len (O0O0OOOO00O0O0OOO )#line:99
    O0O0OOO00OO000OOO .weight =torch .nn .Parameter (O0O0OOO00OO000OOO .weight .data .clone ()[OOOO00O0OOOOO00OO ])#line:100
    if O0O0OOO00OO000OOO .bias is not None :#line:101
        O0O0OOO00OO000OOO .bias =torch .nn .Parameter (O0O0OOO00OO000OOO .bias .data .clone ()[OOOO00O0OOOOO00OO ])#line:102
    return O0O0OOO00OO000OOO ,O0O0000OOO000OO00 #line:103
def prune_related_linear (OOO00000O0OOO0000 :nn .modules .linear .Linear ,O00OOOOOOOOOO000O :list ,inplace :list =True ,dry_run :list =False ):#line:105
    ""#line:111
    OO0OOO00O000O00O0 =len (O00OOOOOOOOOO000O )*OOO00000O0OOO0000 .weight .shape [0 ]#line:112
    if dry_run :#line:113
        return OOO00000O0OOO0000 ,OO0OOO00O000O00O0 #line:114
    if not inplace :#line:116
        OOO00000O0OOO0000 =deepcopy (OOO00000O0OOO0000 )#line:117
    O0O00000OOO0O0OO0 =[OOO0O000O00OOO00O for OOO0O000O00OOO00O in range (OOO00000O0OOO0000 .in_features )if OOO0O000O00OOO00O not in O00OOOOOOOOOO000O ]#line:118
    OOO00000O0OOO0000 .in_features =OOO00000O0OOO0000 .in_features -len (O00OOOOOOOOOO000O )#line:119
    OOO00000O0OOO0000 .weight =torch .nn .Parameter (OOO00000O0OOO0000 .weight .data .clone ()[:,O0O00000OOO0O0OO0 ])#line:120
    return OOO00000O0OOO0000 ,OO0OOO00O000O00O0 #line:121
def prune_batchnorm (O000OOOOO0O0OO000 :nn .modules .batchnorm ._BatchNorm ,OO000O0OO000O0OO0 :list ,inplace :bool =True ,dry_run :bool =False ):#line:123
    ""#line:129
    OO00O00O0O0O00O0O =len (OO000O0OO000O0OO0 )*(2 if O000OOOOO0O0OO000 .affine else 1 )#line:131
    if dry_run :#line:132
        return O000OOOOO0O0OO000 ,OO00O00O0O0O00O0O #line:133
    if not inplace :#line:135
        O000OOOOO0O0OO000 =deepcopy (O000OOOOO0O0OO000 )#line:136
    OO0OOOOO0O0OO00OO =[OOOOO000000O00OO0 for OOOOO000000O00OO0 in range (O000OOOOO0O0OO000 .num_features )if OOOOO000000O00OO0 not in OO000O0OO000O0OO0 ]#line:138
    O000OOOOO0O0OO000 .num_features =O000OOOOO0O0OO000 .num_features -len (OO000O0OO000O0OO0 )#line:139
    O000OOOOO0O0OO000 .running_mean =O000OOOOO0O0OO000 .running_mean .data .clone ()[OO0OOOOO0O0OO00OO ]#line:140
    O000OOOOO0O0OO000 .running_var =O000OOOOO0O0OO000 .running_var .data .clone ()[OO0OOOOO0O0OO00OO ]#line:141
    if O000OOOOO0O0OO000 .affine :#line:142
        O000OOOOO0O0OO000 .weight =torch .nn .Parameter (O000OOOOO0O0OO000 .weight .data .clone ()[OO0OOOOO0O0OO00OO ])#line:143
        O000OOOOO0O0OO000 .bias =torch .nn .Parameter (O000OOOOO0O0OO000 .bias .data .clone ()[OO0OOOOO0O0OO00OO ])#line:144
    return O000OOOOO0O0OO000 ,OO00O00O0O0O00O0O #line:145
def prune_prelu (OO0O00OO0O0O000O0 :nn .PReLU ,OOOOO0O000O00O0OO :list ,inplace :bool =True ,dry_run :bool =False ):#line:147
    ""#line:153
    OOOO00OOOOO000OO0 =0 if OO0O00OO0O0O000O0 .num_parameters ==1 else len (OOOOO0O000O00O0OO )#line:154
    if dry_run :#line:155
        return OO0O00OO0O0O000O0 ,OOOO00OOOOO000OO0 #line:156
    if not inplace :#line:157
        OO0O00OO0O0O000O0 =deepcopy (OO0O00OO0O0O000O0 )#line:158
    if OO0O00OO0O0O000O0 .num_parameters ==1 :return OO0O00OO0O0O000O0 ,OOOO00OOOOO000OO0 #line:159
    OO0000O0O0OO00000 =[OO0OOO000OOOOO000 for OO0OOO000OOOOO000 in range (OO0O00OO0O0O000O0 .num_parameters )if OO0OOO000OOOOO000 not in OOOOO0O000O00O0OO ]#line:160
    OO0O00OO0O0O000O0 .num_parameters =OO0O00OO0O0O000O0 .num_parameters -len (OOOOO0O000O00O0OO )#line:161
    OO0O00OO0O0O000O0 .weight =torch .nn .Parameter (OO0O00OO0O0O000O0 .weight .data .clone ()[OO0000O0O0OO00000 ])#line:162
    return OO0O00OO0O0O000O0 ,OOOO00OOOOO000OO0 #line:163
