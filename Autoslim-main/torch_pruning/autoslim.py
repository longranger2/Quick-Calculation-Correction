import torch #line:1
import torch .nn as nn #line:2
import numpy as np #line:3
from itertools import chain #line:4
from .dependency import *#line:5
from .import prune #line:6
import math #line:7
from scipy .spatial import distance #line:8
__all__ =['Autoslim']#line:10
class Autoslim (object ):#line:12
    def __init__ (O0O00O00OOOOO0000 ,OO0O0OO00O0OO00OO ,O00000000OOOOO0O0 ,O000OO0O0O0OO0000 ):#line:13
        O0O00O00OOOOO0000 .model =OO0O0OO00O0OO00OO #line:14
        O0O00O00OOOOO0000 .inputs =O00000000OOOOO0O0 #line:15
        O0O00O00OOOOO0000 .compression_ratio =O000OO0O0O0OO0000 #line:16
        O0O00O00OOOOO0000 .DG =DependencyGraph ()#line:17
        O0O00O00OOOOO0000 .DG .build_dependency (OO0O0OO00O0OO00OO ,example_inputs =O00000000OOOOO0O0 )#line:19
        O0O00O00OOOOO0000 .model_modules =list (OO0O0OO00O0OO00OO .modules ())#line:20
    def index_of_layer (OOO00OOO00O0000OO ):#line:22
        OOO000O0O0O0O00O0 ={}#line:23
        for O0OOOO000OO0OO00O ,O0O000O0OOOO0O000 in enumerate (OOO00OOO00O0000OO .model_modules ):#line:24
            if isinstance (O0O000O0OOOO0O000 ,nn .modules .conv ._ConvNd ):#line:25
                OOO000O0O0O0O00O0 [O0OOOO000OO0OO00O ]=O0O000O0OOOO0O000 #line:26
        return OOO000O0O0O0O00O0 #line:27
    def base_prunging (OO0O000O00O0OO000 ,OO000OO0OO0000O00 ):#line:32
        OOOOOOOOO0O00OO00 ={}#line:33
        for O00000OO0OOO0O0O0 ,O000O0O00O0O0000O in enumerate (OO0O000O00O0OO000 .model_modules ):#line:34
            if isinstance (O000O0O00O0O0000O ,nn .modules .conv ._ConvNd ):#line:35
                OOOOOOOOO0O00OO00 [O00000OO0OOO0O0O0 ]=O000O0O00O0O0000O .out_channels #line:36
        for O00000OO0OOO0O0O0 ,O000O0O00O0O0000O in enumerate (OO0O000O00O0OO000 .model_modules ):#line:37
            if isinstance (O000O0O00O0O0000O ,nn .modules .conv ._ConvNd ):#line:39
                OOOO0000O0000O0O0 =weight .shape [0 ]#line:40
                if isinstance (O000O0O00O0O0000O ,nn .modules .conv ._ConvTransposeMixin ):#line:42
                    OOOO0000O0000O0O0 =weight .shape [1 ]#line:43
                O00O00O0OOO0O00O0 =OO000OO0OO0000O00 (O000O0O00O0O0000O )#line:45
                if O000O0O00O0O0000O .out_channels ==OOOOOOOOO0O00OO00 [O00000OO0OOO0O0O0 ]:#line:47
                    O00OOO0O0O0OO0000 =OO0O000O00O0OO000 .DG .get_pruning_plan (O000O0O00O0O0000O ,prune .prune_conv ,idxs =O00O00O0OOO0O00O0 )#line:48
                    if O00OOO0O0O0OO0000 :#line:49
                        if prune_shortcut ==1 :#line:50
                            O00OOO0O0O0OO0000 .exec ()#line:51
                    else :#line:52
                        if not O00OOO0O0O0OO0000 .is_in_shortcut :#line:53
                            O00OOO0O0O0OO0000 .exec ()#line:54
    def fpgm_pruning (O00O00OO0OO0OO0O0 ,norm_rate =1 ,dist_type ='l2',layer_compression_ratio =None ,prune_shortcut =1 ):#line:56
        OOO00O0OO0O00OOO0 ={}#line:58
        for O0OO00O0O000OO0OO ,OO00OO0O0OOO0000O in enumerate (O00O00OO0OO0OO0O0 .model_modules ):#line:59
            if isinstance (OO00OO0O0OOO0000O ,nn .modules .conv ._ConvNd ):#line:60
                OOO00O0OO0O00OOO0 [O0OO00O0O000OO0OO ]=OO00OO0O0OOO0000O .out_channels #line:61
        if layer_compression_ratio is None and prune_shortcut ==1 :#line:62
            layer_compression_ratio ={}#line:63
            O0O0O0000OO0O0000 =O00O00OO0OO0OO0O0 .compression_ratio #line:64
            OO0OOOO00O0O0OOOO =(1 -O0O0O0000OO0O0000 )/4 #line:65
            O0O0OO0000OO000O0 =[O0O0O0000OO0O0000 -OO0OOOO00O0O0OOOO *3 ,O0O0O0000OO0O0000 -OO0OOOO00O0O0OOOO *2 ,O0O0O0000OO0O0000 -OO0OOOO00O0O0OOOO ,O0O0O0000OO0O0000 ,O0O0O0000OO0O0000 +OO0OOOO00O0O0OOOO ,O0O0O0000OO0O0000 +OO0OOOO00O0O0OOOO *2 ,O0O0O0000OO0O0000 +OO0OOOO00O0O0OOOO *3 ]#line:66
            OOO0O000O00O0OO0O =0 #line:67
            for O0OO00O0O000OO0OO ,OO00OO0O0OOO0000O in enumerate (O00O00OO0OO0OO0O0 .model_modules ):#line:68
                if isinstance (OO00OO0O0OOO0000O ,nn .modules .conv ._ConvNd ):#line:69
                    layer_compression_ratio [O0OO00O0O000OO0OO ]=0 #line:70
                    OOO0O000O00O0OO0O +=1 #line:71
            OOO0OOO0O0O00O000 =OOO0O000O00O0OO0O /7 #line:72
            O0O0O0O0OO000O0O0 =0 #line:73
            for O0OO00O0O000OO0OO ,OO00OO0O0OOO0000O in enumerate (O00O00OO0OO0OO0O0 .model_modules ):#line:74
                if isinstance (OO00OO0O0OOO0000O ,nn .modules .conv ._ConvNd ):#line:75
                    layer_compression_ratio [O0OO00O0O000OO0OO ]=O0O0OO0000OO000O0 [math .floor (O0O0O0O0OO000O0O0 /OOO0OOO0O0O00O000 )]#line:76
                    O0O0O0O0OO000O0O0 +=1 #line:77
        for O0OO00O0O000OO0OO ,OO00OO0O0OOO0000O in enumerate (O00O00OO0OO0OO0O0 .model_modules ):#line:80
            if isinstance (OO00OO0O0OOO0000O ,nn .modules .conv ._ConvNd ):#line:82
                OO0OOOOO0O0O0OOO0 =OO00OO0O0OOO0000O .weight .detach ().cuda ()#line:83
                O0O00O00O000OO000 =OO0OOOOO0O0O0OOO0 .view (OO0OOOOO0O0O0OOO0 .size ()[0 ],-1 )#line:85
                OOO0OO0O00OOOO0OO =OO0OOOOO0O0O0OOO0 .size ()[0 ]#line:86
                if isinstance (OO00OO0O0OOO0000O ,nn .modules .conv ._ConvTransposeMixin ):#line:88
                    O0O00O00O000OO000 =OO0OOOOO0O0O0OOO0 .view (OO0OOOOO0O0O0OOO0 .size ()[1 ],-1 )#line:89
                    OOO0OO0O00OOOO0OO =OO0OOOOO0O0O0OOO0 .size ()[1 ]#line:90
                if layer_compression_ratio and O0OO00O0O000OO0OO in layer_compression_ratio :#line:92
                    O0OO0OOO00O00000O =int (OOO0OO0O00OOOO0OO *layer_compression_ratio [O0OO00O0O000OO0OO ])#line:93
                else :#line:95
                    O0OO0OOO00O00000O =int (OOO0OO0O00OOOO0OO *O00O00OO0OO0OO0O0 .compression_ratio )#line:96
                OOOOOOOO0OO0O00OO =int (OOO0OO0O00OOOO0OO *(1 -norm_rate ))#line:98
                if dist_type =="l2"or "cos":#line:100
                    OOOOO0OOOOOO00OOO =torch .norm (O0O00O00O000OO000 ,2 ,1 )#line:101
                    OO0O0OOOO000000OO =OOOOO0OOOOOO00OOO .cpu ().numpy ()#line:102
                elif dist_type =="l1":#line:103
                    OOOOO0OOOOOO00OOO =torch .norm (O0O00O00O000OO000 ,1 ,1 )#line:104
                    OO0O0OOOO000000OO =OOOOO0OOOOOO00OOO .cpu ().numpy ()#line:105
                OO0O00O00OO0OO00O =[]#line:107
                OOO000OO0000OOOO0 =[]#line:108
                OOO000OO0000OOOO0 =OO0O0OOOO000000OO .argsort ()[OOOOOOOO0OO0O00OO :]#line:109
                OO0O00O00OO0OO00O =OO0O0OOOO000000OO .argsort ()[:OOOOOOOO0OO0O00OO ]#line:110
                O0O0000O0OOO00000 =torch .LongTensor (OOO000OO0000OOOO0 ).cuda ()#line:113
                OO00O00OOO0O0O0O0 =torch .index_select (O0O00O00O000OO000 ,0 ,O0O0000O0OOO00000 ).cpu ().numpy ()#line:115
                if dist_type =="l2"or "l1":#line:118
                    O0OOOO00OOOO00OO0 =distance .cdist (OO00O00OOO0O0O0O0 ,OO00O00OOO0O0O0O0 ,'euclidean')#line:119
                elif dist_type =="cos":#line:120
                    O0OOOO00OOOO00OO0 =1 -distance .cdist (OO00O00OOO0O0O0O0 ,OO00O00OOO0O0O0O0 ,'cosine')#line:121
                O0OOOOOO00OOOOO00 =np .sum (np .abs (O0OOOO00OOOO00OO0 ),axis =0 )#line:125
                O0OO0OO0000OO0O0O =O0OOOOOO00OOOOO00 .argsort ()[O0OO0OOO00O00000O :]#line:128
                O00OOOOOOOOO0O000 =O0OOOOOO00OOOOO00 .argsort ()[:O0OO0OOO00O00000O ]#line:129
                O0OO00O0O0OOOO0OO =[OOO000OO0000OOOO0 [OO00OO0OOOOO0O0O0 ]for OO00OO0OOOOO0O0O0 in O00OOOOOOOOO0O000 ]#line:130
                if OO00OO0O0OOO0000O .out_channels ==OOO00O0OO0O00OOO0 [O0OO00O0O000OO0OO ]:#line:132
                    OOOOO0OO0O0O00O0O =O00O00OO0OO0OO0O0 .DG .get_pruning_plan (OO00OO0O0OOO0000O ,prune .prune_conv ,idxs =O0OO00O0O0OOOO0OO )#line:133
                    if OOOOO0OO0O0O00O0O :#line:134
                        if prune_shortcut ==1 :#line:135
                            OOOOO0OO0O0O00O0O .exec ()#line:136
                    else :#line:137
                        if not OOOOO0OO0O0O00O0O .is_in_shortcut :#line:138
                            OOOOO0OO0O0O00O0O .exec ()#line:139
    def l1_norm_pruning (OO00000O0OO00OOO0 ,global_pruning =False ,layer_compression_ratio =None ,prune_shortcut =1 ):#line:143
        OOO0OO00OOOO0000O ={}#line:145
        for O00O00OOO0O000000 ,OO00O00O00000OO00 in enumerate (OO00000O0OO00OOO0 .model_modules ):#line:146
            if isinstance (OO00O00O00000OO00 ,nn .modules .conv ._ConvNd ):#line:147
                OOO0OO00OOOO0000O [O00O00OOO0O000000 ]=OO00O00O00000OO00 .out_channels #line:148
        if global_pruning :#line:151
            OO0OOOO0OOOO00000 =[]#line:152
            for O00O00OOO0O000000 ,OO00O00O00000OO00 in enumerate (OO00000O0OO00OOO0 .model_modules ):#line:153
                if isinstance (OO00O00O00000OO00 ,nn .modules .conv ._ConvNd ):#line:154
                    O0O0OOO0O00O000OO =OO00O00O00000OO00 .weight .detach ().cpu ().numpy ()#line:155
                    O0O000OO0O0000OO0 =np .sum (np .abs (O0O0OOO0O00O000OO ),axis =(1 ,2 ,3 ))#line:156
                    if isinstance (OO00O00O00000OO00 ,nn .modules .conv ._ConvTransposeMixin ):#line:157
                        O0O000OO0O0000OO0 =np .sum (np .abs (O0O0OOO0O00O000OO ),axis =(0 ,2 ,3 ))#line:158
                    OO0OOOO0OOOO00000 .append (O0O000OO0O0000OO0 .tolist ())#line:159
            OO0OOOO0OOOO00000 =list (chain .from_iterable (OO0OOOO0OOOO00000 ))#line:161
            O0OOOOOO0OO00OOOO =len (OO0OOOO0OOOO00000 )#line:162
            OO0OOOO0OOOO00000 .sort ()#line:163
            OO00O00OOOO0OOO0O =int (O0OOOOOO0OO00OOOO *OO00000O0OO00OOO0 .compression_ratio )#line:164
            O0OO00O00O000OOO0 =OO0OOOO0OOOO00000 [OO00O00OOOO0OOO0O ]#line:165
            for O00O00OOO0O000000 ,OO00O00O00000OO00 in enumerate (OO00000O0OO00OOO0 .model_modules ):#line:167
                if isinstance (OO00O00O00000OO00 ,nn .modules .conv ._ConvNd ):#line:168
                    O0O0OOO0O00O000OO =OO00O00O00000OO00 .weight .detach ().cpu ().numpy ()#line:169
                    O0O000OO0O0000OO0 =np .sum (np .abs (O0O0OOO0O00O000OO ),axis =(1 ,2 ,3 ))#line:170
                    if isinstance (OO00O00O00000OO00 ,nn .modules .conv ._ConvTransposeMixin ):#line:172
                        O0O000OO0O0000OO0 =np .sum (np .abs (O0O0OOO0O00O000OO ),axis =(0 ,2 ,3 ))#line:173
                    OO0OO0O00O000OO00 =len (O0O000OO0O0000OO0 [O0O000OO0O0000OO0 <O0OO00O00O000OOO0 ])#line:176
                    O0O0O0OO0O0O0O0O0 =np .argsort (O0O000OO0O0000OO0 )[:OO0OO0O00O000OO00 ].tolist ()#line:178
                    if OO00O00O00000OO00 .out_channels ==OOO0OO00OOOO0000O [O00O00OOO0O000000 ]:#line:179
                        OOO0O0000OOO0O0OO =OO00000O0OO00OOO0 .DG .get_pruning_plan (OO00O00O00000OO00 ,prune .prune_conv ,idxs =O0O0O0OO0O0O0O0O0 )#line:180
                        if OOO0O0000OOO0O0OO :#line:181
                            OOO0O0000OOO0O0OO .exec ()#line:182
        else :#line:185
            '''
            自定义压缩时：

            剪跳连层与不剪都可以

            全自动化压缩时：

            剪跳连层：分级剪枝
            不剪跳连层：按照用户指定的阈值剪枝

            '''#line:196
            if layer_compression_ratio is None and prune_shortcut ==1 :#line:199
                layer_compression_ratio ={}#line:200
                O0000O0O00OO00O0O =OO00000O0OO00OOO0 .compression_ratio #line:201
                O0O000000O0O0000O =(1 -O0000O0O00OO00O0O )/4 #line:202
                OOO0O000OOO000O00 =[O0000O0O00OO00O0O -O0O000000O0O0000O *3 ,O0000O0O00OO00O0O -O0O000000O0O0000O *2 ,O0000O0O00OO00O0O -O0O000000O0O0000O ,O0000O0O00OO00O0O ,O0000O0O00OO00O0O +O0O000000O0O0000O ,O0000O0O00OO00O0O +O0O000000O0O0000O *2 ,O0000O0O00OO00O0O +O0O000000O0O0000O *3 ]#line:203
                O0000OOOO00OO00O0 =0 #line:204
                for O00O00OOO0O000000 ,OO00O00O00000OO00 in enumerate (OO00000O0OO00OOO0 .model_modules ):#line:205
                    if isinstance (OO00O00O00000OO00 ,nn .modules .conv ._ConvNd ):#line:206
                        layer_compression_ratio [O00O00OOO0O000000 ]=0 #line:208
                        O0000OOOO00OO00O0 +=1 #line:209
                O00OOOO00000OOOOO =O0000OOOO00OO00O0 /7 #line:210
                OO0OOO00OO0OOO000 =0 #line:211
                for O00O00OOO0O000000 ,OO00O00O00000OO00 in enumerate (OO00000O0OO00OOO0 .model_modules ):#line:212
                    if isinstance (OO00O00O00000OO00 ,nn .modules .conv ._ConvNd ):#line:213
                        layer_compression_ratio [O00O00OOO0O000000 ]=OOO0O000OOO000O00 [math .floor (OO0OOO00OO0OOO000 /O00OOOO00000OOOOO )]#line:214
                        OO0OOO00OO0OOO000 +=1 #line:215
            for O00O00OOO0O000000 ,OO00O00O00000OO00 in enumerate (OO00000O0OO00OOO0 .model_modules ):#line:218
                if isinstance (OO00O00O00000OO00 ,nn .modules .conv ._ConvNd ):#line:220
                    O0O0OOO0O00O000OO =OO00O00O00000OO00 .weight .detach ().cpu ().numpy ()#line:222
                    OO0O000OOOOOO0O00 =O0O0OOO0O00O000OO .shape [0 ]#line:223
                    O0O000OO0O0000OO0 =np .sum (np .abs (O0O0OOO0O00O000OO ),axis =(1 ,2 ,3 ))#line:224
                    if isinstance (OO00O00O00000OO00 ,nn .modules .conv ._ConvTransposeMixin ):#line:226
                        O0O000OO0O0000OO0 =np .sum (np .abs (O0O0OOO0O00O000OO ),axis =(0 ,2 ,3 ))#line:227
                        OO0O000OOOOOO0O00 =O0O0OOO0O00O000OO .shape [1 ]#line:228
                    if layer_compression_ratio and O00O00OOO0O000000 in layer_compression_ratio :#line:231
                        OO0OO0O00O000OO00 =int (OO0O000OOOOOO0O00 *layer_compression_ratio [O00O00OOO0O000000 ])#line:232
                    else :#line:234
                        OO0OO0O00O000OO00 =int (OO0O000OOOOOO0O00 *OO00000O0OO00OOO0 .compression_ratio )#line:235
                    O0O0O0OO0O0O0O0O0 =np .argsort (O0O000OO0O0000OO0 )[:OO0OO0O00O000OO00 ].tolist ()#line:237
                    if OO00O00O00000OO00 .out_channels ==OOO0OO00OOOO0000O [O00O00OOO0O000000 ]:#line:239
                        OOO0O0000OOO0O0OO =OO00000O0OO00OOO0 .DG .get_pruning_plan (OO00O00O00000OO00 ,prune .prune_conv ,idxs =O0O0O0OO0O0O0O0O0 )#line:241
                        if OOO0O0000OOO0O0OO :#line:242
                            if prune_shortcut ==1 :#line:243
                                OOO0O0000OOO0O0OO .exec ()#line:244
                            else :#line:245
                                if not OOO0O0000OOO0O0OO .is_in_shortcut :#line:246
                                    OOO0O0000OOO0O0OO .exec ()#line:247
if __name__ =="__main__":#line:250
    from resnet_small import resnet_small #line:251
    OO0O0OO0O0O00O0OO =resnet_small ()#line:252
    OO0O0OO000O000OOO =Autoslim (OO0O0OO0O0O00O0OO ,inputs =torch .randn (1 ,3 ,224 ,224 ),compression_ratio =0.5 )#line:253
    OO0O0OO000O000OOO .l1_norm_pruning ()#line:254
    print (OO0O0OO0O0O00O0OO )#line:255
