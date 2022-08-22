import torch #line:1
import torch .nn as nn #line:2
import typing #line:3
from functools import reduce #line:4
from operator import mul #line:5
from .import prune #line:6
from enum import IntEnum #line:7
__all__ =['PruningPlan','Dependency','DependencyGraph']#line:9
O0OOO00O0O0OOO0O0 =nn .modules .conv ._ConvNd #line:11
O0OOOOOOO0O0O0OO0 =nn .modules .batchnorm ._BatchNorm #line:12
O0OOO00O0O0O0O0O0 =nn .PReLU #line:13
OO000OOOO000O0OOO =nn .Linear #line:14
class OOOOO0OOOO0O0O0O0 (IntEnum ):#line:16
    CONV =0 #line:17
    BN =1 #line:18
    LINEAR =2 #line:19
    PRELU =3 #line:20
    GROUP_CONV =4 #line:21
    CONCAT =5 #line:23
    SPLIT =6 #line:24
    ELEMENTWISE =7 #line:25
def _O0000O00000000O00 (O00000OO0OO00O000 ):#line:27
    if isinstance (O00000OO0OO00O000 ,O0OOO00O0O0OOO0O0 ):#line:28
        if O00000OO0OO00O000 .groups >1 :#line:29
            return OOOOO0OOOO0O0O0O0 .GROUP_CONV #line:30
        else :#line:31
            return OOOOO0OOOO0O0O0O0 .CONV #line:32
    elif isinstance (O00000OO0OO00O000 ,O0OOOOOOO0O0O0OO0 ):#line:33
        return OOOOO0OOOO0O0O0O0 .BN #line:34
    elif isinstance (O00000OO0OO00O000 ,O0OOO00O0O0O0O0O0 ):#line:35
        return OOOOO0OOOO0O0O0O0 .PRELU #line:36
    elif isinstance (O00000OO0OO00O000 ,OO000OOOO000O0OOO ):#line:37
        return OOOOO0OOOO0O0O0O0 .LINEAR #line:38
    elif isinstance (O00000OO0OO00O000 ,_O0O0O0OO0O0OOO0OO ):#line:39
        return OOOOO0OOOO0O0O0O0 .CONCAT #line:40
    elif isinstance (O00000OO0OO00O000 ,_O00O0O0OO0OO0OOO0 ):#line:41
        return OOOOO0OOOO0O0O0O0 .SPLIT #line:42
    else :#line:43
        return OOOOO0OOOO0O0O0O0 .ELEMENTWISE #line:44
def _O000O0OO0O0O0O0O0 (O0O00OOOOO0000O00 ):#line:46
    if O0O00OOOOO0000O00 .type ==OOOOO0OOOO0O0O0O0 .CONV or O0O00OOOOO0000O00 .type ==OOOOO0OOOO0O0O0O0 .GROUP_CONV :#line:47
        return O0O00OOOOO0000O00 .module .out_channels #line:48
    elif O0O00OOOOO0000O00 .type ==OOOOO0OOOO0O0O0O0 .BN :#line:49
        return O0O00OOOOO0000O00 .module .num_features #line:50
    elif O0O00OOOOO0000O00 .type ==OOOOO0OOOO0O0O0O0 .LINEAR :#line:51
        return O0O00OOOOO0000O00 .module .out_features #line:52
    elif O0O00OOOOO0000O00 .type ==OOOOO0OOOO0O0O0O0 .PRELU :#line:53
        if O0O00OOOOO0000O00 .module .num_parameters ==1 :#line:54
            return None #line:55
        else :#line:56
            return O0O00OOOOO0000O00 .module .num_parameters #line:57
    else :#line:58
        return None #line:59
def _O0OOOOO000OOO0000 (O00O0O000O00O0OO0 ):#line:61
    if O00O0O000O00O0OO0 .type ==OOOOO0OOOO0O0O0O0 .CONV or O00O0O000O00O0OO0 .type ==OOOOO0OOOO0O0O0O0 .GROUP_CONV :#line:62
        return O00O0O000O00O0OO0 .module .in_channels #line:63
    elif O00O0O000O00O0OO0 .type ==OOOOO0OOOO0O0O0O0 .BN :#line:64
        return O00O0O000O00O0OO0 .module .num_features #line:65
    elif O00O0O000O00O0OO0 .type ==OOOOO0OOOO0O0O0O0 .LINEAR :#line:66
        return O00O0O000O00O0OO0 .module .in_features #line:67
    elif O00O0O000O00O0OO0 .type ==OOOOO0OOOO0O0O0O0 .PRELU :#line:68
        if O00O0O000O00O0OO0 .module .num_parameters ==1 :#line:69
            return None #line:70
        else :#line:71
            return O00O0O000O00O0OO0 .module .num_parameters #line:72
    else :#line:73
        return None #line:74
def _O00000O0O000O0OOO (O0OO0O0000OOO0000 ,*OOO0O0000000O00O0 ,**O0OO00O000O0OO0OO ):#line:77
    return O0OO0O0000OOO0000 ,0 #line:78
def _OO000O0OOO0OOO000 (O000OO00O0OO0O000 ,*O000O0O0O0OOO0000 ,**OOO00O0OO00OO00O0 ):#line:80
    return O000OO00O0OO0O000 ,0 #line:81
def _OOOOOOOO0O0O0O00O (O00O0OOO00O0OO0OO ,*OOO0OOOO0O00000OO ,**O00OOOO00000OO00O ):#line:83
    return O00O0OOO00O0OO0OO ,0 #line:84
class _O0O0O0OO0O0OOO0OO (nn .Module ):#line:87
    def __init__ (OOOO0O00O0O00OOO0 ):#line:88
        super (_O0O0O0OO0O0OOO0OO ,OOOO0O00O0O00OOO0 ).__init__ ()#line:89
        OOOO0O00O0O00OOO0 .offsets =None #line:90
    def __repr__ (O0OO00O0OO00O00OO ):#line:92
        return "_ConcatOp(%s)"%(O0OO00O0OO00O00OO .offsets )#line:93
class _O00O0O0OO0OO0OOO0 (nn .Module ):#line:95
    def __init__ (O00O00000OO00OOO0 ):#line:96
        super (_O00O0O0OO0OO0OOO0 ,O00O00000OO00OOO0 ).__init__ ()#line:97
        O00O00000OO00OOO0 .offsets =None #line:98
    def __repr__ (O00000OOO0OOOOOOO ):#line:100
        return "_SplitOP(%s)"%(O00000OOO0OOOOOOO .offsets )#line:101
class _OOO0OOO0O0O00000O (nn .Module ):#line:103
    def __init__ (OO0OO000OOOO0OO0O ):#line:104
        super (_OOO0OOO0O0O00000O ,OO0OO000OOOO0OO0O ).__init__ ()#line:105
    def __repr__ (O0000000O000O0O00 ):#line:107
        return "_ElementWiseOp()"#line:108
class _O0O0O0OOO000O0O00 (object ):#line:112
    def __init__ (O0000O000O000OO00 ,stride =1 ,reverse =False ):#line:113
        O0000O000O000OO00 ._stride =stride #line:114
        O0000O000O000OO00 .reverse =reverse #line:115
    def __call__ (O00O000O0O00O000O ,OO0O0OOO0OOOOO00O ):#line:117
        OO00OO00O00OO0OO0 =[]#line:118
        if O00O000O0O00O000O .reverse ==True :#line:119
            for O00000OO0OOOO000O in OO0O0OOO0OOOOO00O :#line:120
                OO00OO00O00OO0OO0 .append (O00000OO0OOOO000O //O00O000O0O00O000O ._stride )#line:121
                OO00OO00O00OO0OO0 =list (set (OO00OO00O00OO0OO0 ))#line:122
        else :#line:123
            for O00000OO0OOOO000O in OO0O0OOO0OOOOO00O :#line:124
                OO00OO00O00OO0OO0 .extend (list (range (O00000OO0OOOO000O *O00O000O0O00O000O ._stride ,(O00000OO0OOOO000O +1 )*O00O000O0O00O000O ._stride )))#line:125
        return OO00OO00O00OO0OO0 #line:126
class _OOOOOO0OO0O00OOO0 (object ):#line:128
    def __init__ (O000000OOO000O0OO ,O000OO000O0O0OO0O ,reverse =False ):#line:129
        O000000OOO000O0OO .offset =O000OO000O0O0OO0O #line:130
        O000000OOO000O0OO .reverse =reverse #line:131
    def __call__ (O0OO00OOO0OOOO0OO ,OOO0O000O0O0OO00O ):#line:133
        if O0OO00OOO0OOOO0OO .reverse ==True :#line:134
            OOOO000O0OOO0OOO0 =[OOOOOOO00000OO000 -O0OO00OOO0OOOO0OO .offset [0 ]for OOOOOOO00000OO000 in OOO0O000O0O0OO00O if (OOOOOOO00000OO000 >=O0OO00OOO0OOOO0OO .offset [0 ]and OOOOOOO00000OO000 <O0OO00OOO0OOOO0OO .offset [1 ])]#line:135
        else :#line:136
            OOOO000O0OOO0OOO0 =[OO00O0OO00O0O0OOO +O0OO00OOO0OOOO0OO .offset [0 ]for OO00O0OO00O0O0OOO in OOO0O000O0O0OO00O ]#line:137
        return OOOO000O0OOO0OOO0 #line:138
class _O000OOOOOO00O0O0O (object ):#line:140
    def __init__ (OO0O0OO0O000O00OO ,O00O00OO0000O00OO ,reverse =False ):#line:141
        OO0O0OO0O000O00OO .offset =O00O00OO0000O00OO #line:142
        OO0O0OO0O000O00OO .reverse =reverse #line:143
    def __call__ (O0OO0OOO000O0OOOO ,O00O00OO000OOOO0O ):#line:145
        if O0OO0OOO000O0OOOO .reverse ==True :#line:146
            O0OO0000O0O000000 =[O00O000O00O0OOOO0 +O0OO0OOO000O0OOOO .offset [0 ]for O00O000O00O0OOOO0 in O00O00OO000OOOO0O ]#line:147
        else :#line:148
            O0OO0000O0O000000 =[O00O000OOO00OOOO0 -O0OO0OOO000O0OOOO .offset [0 ]for O00O000OOO00OOOO0 in O00O00OO000OOOO0O if (O00O000OOO00OOOO0 >=O0OO0OOO000O0OOOO .offset [0 ]and O00O000OOO00OOOO0 <O0OO0OOO000O0OOOO .offset [1 ])]#line:149
        return O0OO0000O0O000000 #line:150
class O00O00OOO0O0OO0O0 (object ):#line:152
    def __init__ (OO0O00OO000OO000O ,OO00000OOOOO000O0 ,OO0000OO00O0O0OOO ,node_name =None ):#line:153
        OO0O00OO000OO000O .module =OO00000OOOOO000O0 #line:154
        OO0O00OO000OO000O .grad_fn =OO0000OO00O0O0OOO #line:155
        OO0O00OO000OO000O .inputs =[]#line:156
        OO0O00OO000OO000O .outputs =[]#line:157
        OO0O00OO000OO000O .dependencies =[]#line:158
        OO0O00OO000OO000O ._node_name =node_name #line:159
        OO0O00OO000OO000O .type =_O0000O00000000O00 (OO00000OOOOO000O0 )#line:160
    @property #line:162
    def node_name (OOOOOO0O00O00O0O0 ):#line:163
        return "%s (%s)"%(OOOOOO0O00O00O0O0 ._node_name ,str (OOOOOO0O00O00O0O0 .module ))if OOOOOO0O00O00O0O0 ._node_name is not None else str (OOOOOO0O00O00O0O0 .module )#line:164
    def add_input (O0O00OO0OO0O000O0 ,O0O0O00OO0OO0O00O ):#line:166
        if O0O0O00OO0OO0O00O not in O0O00OO0OO0O000O0 .inputs :#line:167
            O0O00OO0OO0O000O0 .inputs .append (O0O0O00OO0OO0O00O )#line:168
    def add_output (O0OOO000O000000OO ,O00000OO0O00O00OO ):#line:170
        if O00000OO0O00O00OO not in O0OOO000O000000OO .outputs :#line:171
            O0OOO000O000000OO .outputs .append (O00000OO0O00O00OO )#line:172
    def __repr__ (OOO000OO0OOO0O00O ):#line:174
        return "<Node: (%s, %s)>"%(OOO000OO0OOO0O00O .node_name ,OOO000OO0OOO0O00O .grad_fn )#line:175
    def __str__ (OO0O000OOO000OO00 ):#line:177
        return "<Node: (%s, %s)>"%(OO0O000OOO000OO00 .node_name ,OO0O000OOO000OO00 .grad_fn )#line:178
    def details (OO00O000O00O000OO ):#line:180
        OOOO0O0OOOOO0O00O ="<Node: (%s, %s)>\n"%(OO00O000O00O000OO .node_name ,OO00O000O00O000OO .grad_fn )#line:181
        OOOO0O0OOOOO0O00O +=' '*4 +'IN:\n'#line:182
        for OOO000OOO0OOO0O00 in OO00O000O00O000OO .inputs :#line:183
            OOOO0O0OOOOO0O00O +=' '*8 +'%s\n'%(OOO000OOO0OOO0O00 )#line:184
        OOOO0O0OOOOO0O00O +=' '*4 +'OUT:\n'#line:185
        for OOO0OO00OOOO0O0OO in OO00O000O00O000OO .outputs :#line:186
            OOOO0O0OOOOO0O00O +=' '*8 +'%s\n'%(OOO0OO00OOOO0O0OO )#line:187
        OOOO0O0OOOOO0O00O +=' '*4 +'DEP:\n'#line:189
        for OO0OOO00000OOO000 in OO00O000O00O000OO .dependencies :#line:190
            OOOO0O0OOOOO0O00O +=' '*8 +"%s\n"%(OO0OOO00000OOO000 )#line:191
        return OOOO0O0OOOOO0O00O #line:192
class Dependency (object ):#line:194
    def __init__ (O0O00000O00OOO0OO ,OOO0O00O00O00O000 ,OO0OOOO0OOOOO0O0O ,OOO0O0O0OO0OO000O :O00O00OOO0O0OO0O0 ,index_transform :typing .Callable =None ):#line:195
        ""#line:202
        O0O00000O00OOO0OO .trigger =OOO0O00O00O00O000 #line:203
        O0O00000O00OOO0OO .handler =OO0OOOO0OOOOO0O0O #line:204
        O0O00000O00OOO0OO .broken_node =OOO0O0O0OO0OO000O #line:205
        O0O00000O00OOO0OO .index_transform =index_transform #line:206
    def __call__ (OOOOOO0OOOOOOO000 ,OOO000OOOOO00OOOO :list ,dry_run :bool =False ):#line:208
        OO0O0OOO0OO0OOOO0 =OOOOOO0OOOOOOO000 .handler (OOOOOO0OOOOOOO000 .broken_node .module ,OOO000OOOOO00OOOO ,dry_run =dry_run )#line:209
        return OO0O0OOO0OO0OOOO0 #line:210
    def __repr__ (O00OOO0000O0OOO00 ):#line:212
        return str (O00OOO0000O0OOO00 )#line:213
    def __str__ (OOO0OO0O000O000O0 ):#line:215
        return "<DEP: %s => %s on %s>"%("None"if OOO0OO0O000O000O0 .trigger is None else OOO0OO0O000O000O0 .trigger .__name__ ,OOO0OO0O000O000O0 .handler .__name__ ,OOO0OO0O000O000O0 .broken_node .node_name )#line:216
    def is_triggered_by (O0O000OO0OO000OOO ,OO0O0000OOO000OO0 ):#line:218
        return OO0O0000OOO000OO0 ==O0O000OO0OO000OOO .trigger #line:219
    def __eq__ (O00O00OO0OO00O0OO ,OOO00OO000OO0000O ):#line:221
        return ((O00O00OO0OO00O0OO .trigger ==OOO00OO000OO0000O .trigger )and O00O00OO0OO00O0OO .handler ==OOO00OO000OO0000O .handler and O00O00OO0OO00O0OO .broken_node ==OOO00OO000OO0000O .broken_node )#line:224
class PruningPlan (object ):#line:226
    ""#line:232
    def __init__ (OO000OO00OOOO0000 ):#line:234
        OO000OO00OOOO0000 ._plans =list ()#line:235
    def add_plan (O0000O00OO0OOOOOO ,O00OOO00OOO00O000 ,OO00O000OOOOO0O0O ):#line:237
        O0000O00OO0OOOOOO ._plans .append ((O00OOO00OOO00O000 ,OO00O000OOOOO0O0O ))#line:238
    @property #line:240
    def plan (OO00O0O00O00000O0 ):#line:241
        return OO00O0O00O00000O0 ._plans #line:242
    def exec (O00O0OOO000O00000 ,dry_run =False ):#line:244
        O000OOOO0OO0OOO00 =0 #line:245
        for OOO0OO0OOOOOOO0OO ,O00O0O0O00000OOOO in O00O0OOO000O00000 ._plans :#line:246
            _OO0O000OOO00OO000 ,O0000O00OOOOO000O =OOO0OO0OOOOOOO0OO (O00O0O0O00000OOOO ,dry_run =dry_run )#line:247
            O000OOOO0OO0OOO00 +=O0000O00OOOOO000O #line:248
        return O000OOOO0OO0OOO00 #line:249
    def has_dep (O0OO0O0O000OO00O0 ,OO00O0O0OOOOO0OOO ):#line:251
        for _OO000OOOO00O0O0O0 ,_O000O00O00OOO0OOO in O0OO0O0O000OO00O0 ._plans :#line:252
            if OO00O0O0OOOOO0OOO ==_OO000OOOO00O0O0O0 :#line:253
                return True #line:254
        return False #line:255
    def has_pruning_op (O0O0O0000OOOOOO00 ,OOOO00O00OO0OO0OO ,OOO0OOOO0O00O0OO0 ):#line:257
        for _O00OOOO0O00OO00OO ,_O0OOOO0O0OO00O00O in O0O0O0000OOOOOO00 ._plans :#line:258
            if _O00OOOO0O00OO00OO .broken_node ==OOOO00O00OO0OO0OO .broken_node and _O00OOOO0O00OO00OO .handler ==OOOO00O00OO0OO0OO .handler and _O0OOOO0O0OO00O00O ==OOO0OOOO0O00O0OO0 :#line:259
                return True #line:260
        return False #line:261
    @property #line:263
    def is_in_shortcut (O00OOOOOOO000OO0O ):#line:264
        OOO0O0OO00O00O0OO =0 #line:265
        for _O00OOOOO0OOO0O0O0 ,_OOO00O0O000OOO0O0 in O00OOOOOOO000OO0O ._plans :#line:266
            if _O00OOOOO0OOO0O0O0 .handler .__name__ =='prune_conv':#line:267
                OOO0O0OO00O00O0OO +=1 #line:268
        if OOO0O0OO00O00O0OO >1 :#line:269
            return True #line:270
        else :#line:271
            return False #line:272
    def add_plan_and_merge (OOOOO0O00OOO00O0O ,O0OO00OOO0OO00O0O ,OOOOO0000O000OOO0 ):#line:274
        for OO0OOOOOOO0OOO0OO ,(_O0O0OOO00OO0O0O00 ,_O0000000OO0O00O00 )in enumerate (OOOOO0O00OOO00O0O ._plans ):#line:275
            if _O0O0OOO00OO0O0O00 .broken_node ==O0OO00OOO0OO00O0O .broken_node and _O0O0OOO00OO0O0O00 .handler ==O0OO00OOO0OO00O0O .handler :#line:276
                OOOOO0O00OOO00O0O ._plans [OO0OOOOOOO0OOO0OO ]=(_O0O0OOO00OO0O0O00 ,list (set (_O0000000OO0O00O00 +OOOOO0000O000OOO0 )))#line:277
                return #line:278
        OOOOO0O00OOO00O0O .add_plan (O0OO00OOO0OO00O0O ,OOOOO0000O000OOO0 )#line:279
    def __str__ (O00000O0O0OOO00O0 ):#line:281
        OO000O00000O0O0O0 =""#line:282
        OO000O00000O0O0O0 +="\n-------------\n"#line:283
        OOO0O0O0O00OO00OO =0 #line:284
        for OO00OO0OOOO00O000 ,OO0O0OO0O00OOO0OO in O00000O0O0OOO00O0 ._plans :#line:285
            _O000O00O00O00O00O ,OOOO0OOO0000O00OO =OO00OO0OOOO00O000 (OO0O0OO0O00OOO0OO ,dry_run =True )#line:286
            OOO0O0O0O00OO00OO +=OOOO0OOO0000O00OO #line:287
            OO000O00000O0O0O0 +="[ %s, Index=%s, NumPruned=%d]\n"%(OO00OO0OOOO00O000 ,OO0O0OO0O00OOO0OO ,OOOO0OOO0000O00OO )#line:288
        OO000O00000O0O0O0 +="%d parameters will be pruned\n"%(OOO0O0O0O00OO00OO )#line:289
        OO000O00000O0O0O0 +="-------------\n"#line:290
        return OO000O00000O0O0O0 #line:291
class DependencyGraph (object ):#line:294
    PRUNABLE_MODULES =(nn .modules .conv ._ConvNd ,nn .modules .batchnorm ._BatchNorm ,nn .Linear ,nn .PReLU )#line:296
    HANDLER ={OOOOO0OOOO0O0O0O0 .CONV :(prune .prune_related_conv ,prune .prune_conv ),OOOOO0OOOO0O0O0O0 .BN :(prune .prune_batchnorm ,prune .prune_batchnorm ),OOOOO0OOOO0O0O0O0 .PRELU :(prune .prune_prelu ,prune .prune_prelu ),OOOOO0OOOO0O0O0O0 .LINEAR :(prune .prune_related_linear ,prune .prune_linear ),OOOOO0OOOO0O0O0O0 .GROUP_CONV :(prune .prune_group_conv ,prune .prune_group_conv ),OOOOO0OOOO0O0O0O0 .CONCAT :(_O00000O0O000O0OOO ,_O00000O0O000O0OOO ),OOOOO0OOOO0O0O0O0 .SPLIT :(_OO000O0OOO0OOO000 ,_OO000O0OOO0OOO000 ),OOOOO0OOOO0O0O0O0 .ELEMENTWISE :(_OOOOOOOO0O0O0O00O ,_OOOOOOOO0O0O0O00O ),}#line:307
    OUTPUT_NODE_RULES ={}#line:308
    INPUT_NODE_RULES ={}#line:309
    for t1 in HANDLER .keys ():#line:310
        for t2 in HANDLER .keys ():#line:311
            OUTPUT_NODE_RULES [(t1 ,t2 )]=(HANDLER [t1 ][1 ],HANDLER [t2 ][0 ])#line:312
            INPUT_NODE_RULES [(t1 ,t2 )]=(HANDLER [t1 ][0 ],HANDLER [t2 ][1 ])#line:313
    def build_dependency (O000O0OOO00O000O0 ,OOOOO0OOOOO0000OO :torch .nn .Module ,OOO0OO0OOO0OOOO00 :torch .Tensor ,output_transform :callable =None ,verbose :bool =True ):#line:315
        O000O0OOO00O000O0 .verbose =verbose #line:316
        O000O0OOO00O000O0 ._module_to_name ={O0000OO00OOO0OOOO :OOO0O0000OOOO00OO for (OOO0O0000OOOO00OO ,O0000OO00OOO0OOOO )in OOOOO0OOOOO0000OO .named_modules ()}#line:318
        O000O0OOO00O000O0 .module_to_node ,O000O0OOO00O000O0 .output_grad_fn =O000O0OOO00O000O0 ._obtain_forward_graph (OOOOO0OOOOO0000OO ,OOO0OO0OOO0OOOO00 ,output_transform =output_transform )#line:320
        O000O0OOO00O000O0 ._build_dependency (O000O0OOO00O000O0 .module_to_node )#line:321
        O000O0OOO00O000O0 .update_index ()#line:322
        return O000O0OOO00O000O0 #line:323
    def update_index (O0OOO0O000OO0O000 ):#line:325
        for OOO0000OOO0O0OOOO ,O0OOO0OOO00000OO0 in O0OOO0O000OO0O000 .module_to_node .items ():#line:326
            if O0OOO0OOO00000OO0 .type ==OOOOO0OOOO0O0O0O0 .LINEAR :#line:327
                O0OOO0O000OO0O000 ._set_fc_index_transform (O0OOO0OOO00000OO0 )#line:328
            if O0OOO0OOO00000OO0 .type ==OOOOO0OOOO0O0O0O0 .CONCAT :#line:329
                O0OOO0O000OO0O000 ._set_concat_index_transform (O0OOO0OOO00000OO0 )#line:330
            if O0OOO0OOO00000OO0 .type ==OOOOO0OOOO0O0O0O0 .SPLIT :#line:331
                O0OOO0O000OO0O000 ._set_split_index_transform (O0OOO0OOO00000OO0 )#line:332
    def get_pruning_plan (O000O0O000O0O00OO ,O0O0OOOO000O000OO ,O000000OO000OO0OO ,O00O0O0OOOO0OO0OO ):#line:334
        if isinstance (O0O0OOOO000O000OO ,O0OOO00O0O0OOO0O0 )and O0O0OOOO000O000OO .groups >1 :#line:335
            O000000OO000OO0OO =prune .prune_group_conv #line:336
        O000O0O000O0O00OO .update_index ()#line:338
        OOOO0O0OOOOO0OO0O =PruningPlan ()#line:339
        OOO0OO0OO0OOOOOO0 =O000O0O000O0O00OO .module_to_node [O0O0OOOO000O000OO ]#line:341
        if OOO0OO0OO0OOOOOO0 .grad_fn in O000O0O000O0O00OO .output_grad_fn :#line:343
            return None #line:344
        OOOO0O0OOOOO0OO0O .add_plan (Dependency (O000000OO000OO0OO ,O000000OO000OO0OO ,OOO0OO0OO0OOOOOO0 ),O00O0O0OOOO0OO0OO )#line:346
        OOO0O000O0OO00OO0 =set ()#line:348
        def _OO000OOO0O00O000O (O0OO0OO0000O0000O ,O0OOOOOOO0OOOOOO0 ,OO00OO0000O0OO0O0 ):#line:349
            OOO0O000O0OO00OO0 .add (O0OO0OO0000O0000O )#line:350
            for OO000OOO00000000O in O0OO0OO0000O0000O .dependencies :#line:351
                if OO000OOO00000000O .is_triggered_by (O0OOOOOOO0OOOOOO0 ):#line:352
                    if OO000OOO00000000O .index_transform is not None :#line:353
                        OO0OO0OOOOO0O0O00 =OO000OOO00000000O .index_transform (OO00OO0000O0OO0O0 )#line:354
                    else :#line:355
                        OO0OO0OOOOO0O0O00 =OO00OO0000O0OO0O0 #line:356
                    if len (OO0OO0OOOOO0O0O00 )==0 :#line:358
                        continue #line:359
                    if OO000OOO00000000O .broken_node in OOO0O000O0OO00OO0 and OOOO0O0OOOOO0OO0O .has_pruning_op (OO000OOO00000000O ,OO0OO0OOOOO0O0O00 ):#line:360
                        continue #line:361
                    else :#line:362
                        OOOO0O0OOOOO0OO0O .add_plan (OO000OOO00000000O ,OO0OO0OOOOO0O0O00 )#line:363
                        _OO000OOO0O00O000O (OO000OOO00000000O .broken_node ,OO000OOO00000000O .handler ,OO0OO0OOOOO0O0O00 )#line:364
        _OO000OOO0O00O000O (OOO0OO0OO0OOOOOO0 ,O000000OO000OO0OO ,O00O0O0OOOO0OO0OO )#line:366
        O0O0O0OOOOO0000OO =PruningPlan ()#line:369
        for OOOO0OO0O00O0OOO0 ,O00O0O0OOOO0OO0OO in OOOO0O0OOOOO0OO0O .plan :#line:370
            O0O0O0OOOOO0000OO .add_plan_and_merge (OOOO0OO0O00O0OOO0 ,O00O0O0OOOO0OO0OO )#line:371
        return O0O0O0OOOOO0000OO #line:372
    def _build_dependency (O00O000O0OOOOO0OO ,O0OOO00O0O0000O0O ):#line:374
        for OO0000O00OO0O00OO ,OOOO0O0OOOO00OO0O in O0OOO00O0O0000O0O .items ():#line:375
            for OO0OO0OOO00O0OOOO in OOOO0O0OOOO00OO0O .inputs :#line:376
                O0O0000O0O00O0O0O =O00O000O0OOOOO0OO .INPUT_NODE_RULES .get ((OOOO0O0OOOO00OO0O .type ,OO0OO0OOO00O0OOOO .type ),None )#line:377
                if O0O0000O0O00O0O0O is not None :#line:378
                    OOO0OO0O00OOOO00O =Dependency (trigger =O0O0000O0O00O0O0O [0 ],handler =O0O0000O0O00O0O0O [1 ],broken_node =OO0OO0OOO00O0OOOO )#line:379
                    OOOO0O0OOOO00OO0O .dependencies .append (OOO0OO0O00OOOO00O )#line:380
            for OO00OOO0O0OOOOO0O in OOOO0O0OOOO00OO0O .outputs :#line:382
                OO0O00O0O00O0OOOO =O00O000O0OOOOO0OO .OUTPUT_NODE_RULES .get ((OOOO0O0OOOO00OO0O .type ,OO00OOO0O0OOOOO0O .type ),None )#line:383
                if OO0O00O0O00O0OOOO is not None :#line:384
                    OOO0OO0O00OOOO00O =Dependency (trigger =OO0O00O0O00O0OOOO [0 ],handler =OO0O00O0O00O0OOOO [1 ],broken_node =OO00OOO0O0OOOOO0O )#line:385
                    OOOO0O0OOOO00OO0O .dependencies .append (OOO0OO0O00OOOO00O )#line:386
    def _obtain_forward_graph (O00OO0000OOO0O0O0 ,OO0OOO000O0O0000O ,OOO000O000O00O0OO ,OOO0O00000OO000O0 ):#line:388
        OO0OOO000O0O0000O .eval ().cpu ()#line:390
        OOOO0000O000000OO ={}#line:392
        O00O0OO0O000O00O0 ={}#line:394
        def _O00O0OO000O0OOO0O (O00000O0000000O00 ,OOO00OO0OO00000OO ,OO000OOOO000O0O00 ):#line:395
            if O00000O0000000O00 not in O00O0OO0O000O00O0 :#line:396
                O00O0OO0O000O00O0 [O00000O0000000O00 ]=1 #line:397
            else :#line:398
                O00O0OO0O000O00O0 [O00000O0000000O00 ]+=1 #line:399
            OOOO0000O000000OO [OO000OOOO000O0O00 .grad_fn ]=O00000O0000000O00 #line:400
        O00OOO0O00O0000O0 =[OOO0000OO00OO000O .register_forward_hook (_O00O0OO000O0OOO0O )for OOO0000OO00OO000O in OO0OOO000O0O0000O .modules ()if isinstance (OOO0000OO00OO000O ,O00OO0000OOO0O0O0 .PRUNABLE_MODULES )]#line:402
        O00O00O00000OOO0O =OO0OOO000O0O0000O (OOO000O000O00O0OO )#line:403
        for O000OOO00OO0OOOOO in O00OOO0O00O0000O0 :#line:404
            O000OOO00OO0OOOOO .remove ()#line:405
        OO000O0OO0O0O00OO =[O00OO00O0O000OO00 for (O00OO00O0O000OO00 ,O000O0O00000O0OOO )in O00O0OO0O000O00O0 .items ()if O000O0O00000O0OOO >1 ]#line:406
        OO0000OOO0O000OO0 ={}#line:408
        OOOO000O000OO000O =[]#line:410
        def _O0O0OOOO0OO00O0O0 (O0OO0OOO0000000OO ,search_final_conv =0 ):#line:412
            search_final_conv =search_final_conv #line:414
            O0O000OOOO000O000 =OOOO0000O000000OO .get (O0OO0OOO0000000OO ,None )#line:416
            if O0O000OOOO000O000 is not None and O0O000OOOO000O000 in OO0000OOO0O000OO0 and O0O000OOOO000O000 not in OO000O0OO0O0O00OO :#line:417
                return OO0000OOO0O000OO0 [O0O000OOOO000O000 ]#line:418
            if O0O000OOOO000O000 is None :#line:420
                if not hasattr (O0OO0OOO0000000OO ,'name'):#line:421
                    O0O000OOOO000O000 =_OOO0OOO0O0O00000O ()#line:422
                    if O00OO0000OOO0O0O0 .verbose :#line:423
                        print ("[Warning] Unrecognized operation: %s. It will be treated as element-wise op"%(str (O0OO0OOO0000000OO )))#line:424
                elif 'catbackward'in O0OO0OOO0000000OO .name ().lower ():#line:425
                    O0O000OOOO000O000 =_O0O0O0OO0O0OOO0OO ()#line:426
                elif 'splitbackward'in O0OO0OOO0000000OO .name ().lower ():#line:427
                    O0O000OOOO000O000 =_O00O0O0OO0OO0OOO0 ()#line:428
                else :#line:429
                    O0O000OOOO000O000 =_OOO0OOO0O0O00000O ()#line:430
                OOOO0000O000000OO [O0OO0OOO0000000OO ]=O0O000OOOO000O000 #line:431
            if O0O000OOOO000O000 not in OO0000OOO0O000OO0 :#line:433
                O0OOOO00O0OOO00OO =O00O00OOO0O0OO0O0 (O0O000OOOO000O000 ,O0OO0OOO0000000OO ,O00OO0000OOO0O0O0 ._module_to_name .get (O0O000OOOO000O000 ,None ))#line:434
                OO0000OOO0O000OO0 [O0O000OOOO000O000 ]=O0OOOO00O0OOO00OO #line:435
            else :#line:436
                O0OOOO00O0OOO00OO =OO0000OOO0O000OO0 [O0O000OOOO000O000 ]#line:437
            if search_final_conv and O0OO0OOO0000000OO is not None and hasattr (O0OO0OOO0000000OO ,'name')and ('MkldnnConvolutionBackward'in O0OO0OOO0000000OO .name ()or 'AddmmBackward'in O0OO0OOO0000000OO .name ()):#line:439
                search_final_conv =0 #line:440
                OOOO000O000OO000O .append (O0OO0OOO0000000OO )#line:441
            if hasattr (O0OO0OOO0000000OO ,'next_functions'):#line:445
                for OO00O0O0O0OO0O0O0 in O0OO0OOO0000000OO .next_functions :#line:446
                    if OO00O0O0O0OO0O0O0 [0 ]is not None :#line:447
                        if hasattr (OO00O0O0O0OO0O0O0 [0 ],'name')and 'accumulategrad'in OO00O0O0O0OO0O0O0 [0 ].name ().lower ():#line:448
                            continue #line:449
                        O0OO0O0O0000O00OO =_O0O0OOOO0OO00O0O0 (OO00O0O0O0OO0O0O0 [0 ],search_final_conv )#line:450
                        O0OOOO00O0OOO00OO .add_input (O0OO0O0O0000O00OO )#line:451
                        O0OO0O0O0000O00OO .add_output (O0OOOO00O0OOO00OO )#line:452
            return O0OOOO00O0OOO00OO #line:453
        if OOO0O00000OO000O0 is not None :#line:455
            O00O00O00000OOO0O =OOO0O00000OO000O0 (O00O00O00000OOO0O )#line:456
        if isinstance (O00O00O00000OOO0O ,(list ,tuple )):#line:458
            for O0OO0OOOOOO0OOOO0 in O00O00O00000OOO0O :#line:460
                if isinstance (O0OO0OOOOOO0OOOO0 ,dict ):#line:461
                    for O00O0O0O000000OO0 in O0OO0OOOOOO0OOOO0 :#line:462
                        if O0OO0OOOOOO0OOOO0 [O00O0O0O000000OO0 ].grad_fn is not None and hasattr (O0OO0OOOOOO0OOOO0 [O00O0O0O000000OO0 ].grad_fn ,'name')and ('MkldnnConvolutionBackward'in O0OO0OOOOOO0OOOO0 [O00O0O0O000000OO0 ].grad_fn .name ()or 'AddmmBackward'in O0OO0OOOOOO0OOOO0 [O00O0O0O000000OO0 ].grad_fn .name ()):#line:464
                            OOOO000O000OO000O .append (O0OO0OOOOOO0OOOO0 [O00O0O0O000000OO0 ].grad_fn )#line:465
                            _O0O0OOOO0OO00O0O0 (O0OO0OOOOOO0OOOO0 [O00O0O0O000000OO0 ].grad_fn ,search_final_conv =0 )#line:466
                        else :#line:467
                            _O0O0OOOO0OO00O0O0 (O0OO0OOOOOO0OOOO0 [O00O0O0O000000OO0 ].grad_fn ,search_final_conv =1 )#line:468
                elif isinstance (O0OO0OOOOOO0OOOO0 ,(list ,tuple )):#line:470
                    for OOOO000OO00O0O00O in O0OO0OOOOOO0OOOO0 :#line:471
                        if OOOO000OO00O0O00O .grad_fn is not None and hasattr (OOOO000OO00O0O00O .grad_fn ,'name')and ('MkldnnConvolutionBackward'in OOOO000OO00O0O00O .grad_fn .name ()or 'AddmmBackward'in OOOO000OO00O0O00O .grad_fn .name ()):#line:472
                            OOOO000O000OO000O .append (OOOO000OO00O0O00O .grad_fn )#line:473
                            _O0O0OOOO0OO00O0O0 (OOOO000OO00O0O00O .grad_fn ,search_final_conv =0 )#line:474
                        else :#line:475
                            _O0O0OOOO0OO00O0O0 (OOOO000OO00O0O00O .grad_fn ,search_final_conv =1 )#line:476
                else :#line:477
                    if O0OO0OOOOOO0OOOO0 .grad_fn is not None and hasattr (O0OO0OOOOOO0OOOO0 .grad_fn ,'name')and ('MkldnnConvolutionBackward'in O0OO0OOOOOO0OOOO0 .grad_fn .name ()or 'AddmmBackward'in O0OO0OOOOOO0OOOO0 .grad_fn .name ()):#line:478
                        OOOO000O000OO000O .append (O0OO0OOOOOO0OOOO0 .grad_fn )#line:479
                        _O0O0OOOO0OO00O0O0 (O0OO0OOOOOO0OOOO0 .grad_fn ,search_final_conv =0 )#line:480
                    else :#line:481
                        _O0O0OOOO0OO00O0O0 (O0OO0OOOOOO0OOOO0 .grad_fn ,search_final_conv =1 )#line:482
        else :#line:485
            if O00O00O00000OOO0O .grad_fn is not None and hasattr (O00O00O00000OOO0O .grad_fn ,'name')and ('MkldnnConvolutionBackward'in O00O00O00000OOO0O .grad_fn .name ()or 'AddmmBackward'in O00O00O00000OOO0O .grad_fn .name ()):#line:486
                OOOO000O000OO000O .append (O00O00O00000OOO0O .grad_fn )#line:487
                _O0O0OOOO0OO00O0O0 (O00O00O00000OOO0O .grad_fn ,search_final_conv =0 )#line:488
            else :#line:489
                _O0O0OOOO0OO00O0O0 (O00O00O00000OOO0O .grad_fn ,search_final_conv =1 )#line:490
        return OO0000OOO0O000OO0 ,OOOO000O000OO000O #line:491
    def _set_fc_index_transform (O0O0O0OOO0000O00O ,OO00OOOOOO00O0O0O :O00O00OOO0O0OO0O0 ):#line:493
        if OO00OOOOOO00O0O0O .type !=OOOOO0OOOO0O0O0O0 .LINEAR :#line:494
            return #line:495
        OOOO0000000000OOO =set ()#line:496
        OOO000O0O000O0000 =OO00OOOOOO00O0O0O .module .in_features #line:497
        OO0OOOOO0OOOO00OO =_O0OOO000OOO0O0OO0 (OO00OOOOOO00O0O0O .inputs [0 ])#line:498
        OO000O0O0O00OO000 =OOO000O0O000O0000 //OO0OOOOO0OOOO00OO #line:499
        if OO000O0O0O00OO000 >1 :#line:500
            for O0OO0OOO0OOO000O0 in OO00OOOOOO00O0O0O .inputs :#line:501
                for OOOOOO00OO0000O00 in OO00OOOOOO00O0O0O .dependencies :#line:502
                    if OOOOOO00OO0000O00 .broken_node ==O0OO0OOO0OOO000O0 :#line:503
                        OOOOOO00OO0000O00 .index_transform =_O0O0O0OOO000O0O00 (stride =OO000O0O0O00OO000 ,reverse =True )#line:504
                for OOOOOO00OO0000O00 in O0OO0OOO0OOO000O0 .dependencies :#line:506
                    if OOOOOO00OO0000O00 .broken_node ==OO00OOOOOO00O0O0O :#line:507
                        OOOOOO00OO0000O00 .index_transform =_O0O0O0OOO000O0O00 (stride =OO000O0O0O00OO000 ,reverse =False )#line:508
    def _set_concat_index_transform (OO00OOOOO00OO00O0 ,O0OOO00OO0000OO00 :O00O00OOO0O0OO0O0 ):#line:510
        if O0OOO00OO0000OO00 .type !=OOOOO0OOOO0O0O0O0 .CONCAT :#line:511
            return #line:512
        OO00OO0O00OO0O000 =[]#line:514
        for OO0O00OOOOO0O0O0O in O0OOO00OO0000OO00 .inputs :#line:515
            OO00OO0O00OO0O000 .append (_O0OOO000OOO0O0OO0 (OO0O00OOOOO0O0O0O ))#line:516
        O0OO0O00O00O0000O =[0 ]#line:518
        for O0OOOO0OOO0000O00 in OO00OO0O00OO0O000 :#line:519
            O0OO0O00O00O0000O .append (O0OO0O00O00O0000O [-1 ]+O0OOOO0OOO0000O00 )#line:520
        O0OOO00OO0000OO00 .module .offsets =O0OO0O00O00O0000O #line:521
        for OOOO0O0OOOO00OO00 ,OO00O00OO00O00OOO in enumerate (O0OOO00OO0000OO00 .inputs ):#line:523
            for OOOOO000OOO0O0000 in O0OOO00OO0000OO00 .dependencies :#line:524
                if OOOOO000OOO0O0000 .broken_node ==OO00O00OO00O00OOO :#line:525
                    OOOOO000OOO0O0000 .index_transform =_OOOOOO0OO0O00OOO0 (offset =O0OO0O00O00O0000O [OOOO0O0OOOO00OO00 :OOOO0O0OOOO00OO00 +2 ],reverse =True )#line:526
            for OOOOO000OOO0O0000 in OO00O00OO00O00OOO .dependencies :#line:528
                if OOOOO000OOO0O0000 .broken_node ==O0OOO00OO0000OO00 :#line:529
                    OOOOO000OOO0O0000 .index_transform =_OOOOOO0OO0O00OOO0 (offset =O0OO0O00O00O0000O [OOOO0O0OOOO00OO00 :OOOO0O0OOOO00OO00 +2 ],reverse =False )#line:530
    def _set_split_index_transform (O0O0OOOO0OO00OO0O ,OOOO0OO00O00O00O0 :O00O00OOO0O0OO0O0 ):#line:532
        if OOOO0OO00O00O00O0 .type !=OOOOO0OOOO0O0O0O0 .SPLIT :#line:533
            return #line:534
        O0OOO00OOOO00O00O =[]#line:536
        for O00OOOOOOO000000O in OOOO0OO00O00O00O0 .outputs :#line:537
            O0OOO00OOOO00O00O .append (_OO0O0O000OOO0OO00 (O00OOOOOOO000000O ))#line:538
        OO000O00O0OOOOOO0 =[0 ]#line:540
        for OO0OO000O000OO000 in O0OOO00OOOO00O00O :#line:541
            OO000O00O0OOOOOO0 .append (OO000O00O0OOOOOO0 [-1 ]+OO0OO000O000OO000 )#line:542
        OOOO0OO00O00O00O0 .module .offsets =OO000O00O0OOOOOO0 #line:543
        for O0O0OO00000O0O000 ,O000OOO0OO00O0000 in enumerate (OOOO0OO00O00O00O0 .outputs ):#line:544
            for OO00OO00O00O0O0OO in OOOO0OO00O00O00O0 .dependencies :#line:545
                if OO00OO00O00O0O0OO .broken_node ==O000OOO0OO00O0000 :#line:546
                    OO00OO00O00O0O0OO .index_transform =_O000OOOOOO00O0O0O (offset =OO000O00O0OOOOOO0 [O0O0OO00000O0O000 :O0O0OO00000O0O000 +2 ],reverse =False )#line:547
            for OO00OO00O00O0O0OO in O000OOO0OO00O0000 .dependencies :#line:549
                if OO00OO00O00O0O0OO .broken_node ==OOOO0OO00O00O00O0 :#line:550
                    OO00OO00O00O0O0OO .index_transform =_O000OOOOOO00O0O0O (offset =OO000O00O0OOOOOO0 [O0O0OO00000O0O000 :O0O0OO00000O0O000 +2 ],reverse =True )#line:551
def _O0OOO000OOO0O0OO0 (O0000O00O000O0O0O ):#line:553
    O000O00OO000OO0O0 =_O000O0OO0O0O0O0O0 (O0000O00O000O0O0O )#line:554
    if O000O00OO000OO0O0 is None :#line:555
        O000O00OO000OO0O0 =0 #line:556
        for OO00OOO00OOO00000 in O0000O00O000O0O0O .inputs :#line:557
            if O0000O00O000O0O0O .type ==OOOOO0OOOO0O0O0O0 .CONCAT :#line:558
                O000O00OO000OO0O0 +=_O0OOO000OOO0O0OO0 (OO00OOO00OOO00000 )#line:559
            else :#line:560
                O000O00OO000OO0O0 =_O0OOO000OOO0O0OO0 (OO00OOO00OOO00000 )#line:561
    return O000O00OO000OO0O0 #line:562
def _OO0O0O000OOO0OO00 (OO0O000OO0O0O0O0O ):#line:564
    O0OOOOO00O0OO0OOO =_O0OOOOO000OOO0000 (OO0O000OO0O0O0O0O )#line:565
    if O0OOOOO00O0OO0OOO is None :#line:566
        O0OOOOO00O0OO0OOO =0 #line:567
        for O00O0OOOOO0OOOO0O in OO0O000OO0O0O0O0O .outputs :#line:568
            if OO0O000OO0O0O0O0O .type ==OOOOO0OOOO0O0O0O0 .SPLIT :#line:569
                O0OOOOO00O0OO0OOO +=_OO0O0O000OOO0OO00 (O00O0OOOOO0OOOO0O )#line:570
            else :#line:571
                O0OOOOO00O0OO0OOO =_OO0O0O000OOO0OO00 (O00O0OOOOO0OOOO0O )#line:572
    return O0OOOOO00O0OO0OOO 