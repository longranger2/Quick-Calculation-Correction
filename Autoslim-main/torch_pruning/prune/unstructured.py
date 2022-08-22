import torch #line:1
import torch .nn as nn #line:2
from copy import deepcopy #line:3
__all__ =['mask_weight','mask_bias']#line:5
def _OO0OO00O0OOO00OO0 (OOOOOOO0O00OO0O00 ,OO00O0OO00O0O0OOO ):#line:7
    if hasattr (OOOOOOO0O00OO0O00 ,'weight_mask'):#line:8
        OOOOOOO0O00OO0O00 .weight .data *=OOOOOOO0O00OO0O00 .weight_mask #line:9
def _O0000OOOO00O000O0 (OO0OOO0OO0O00O000 ,O0O0O0OOOO0OO0O0O ):#line:11
    if OO0OOO0OO0O00O000 .bias is not None and hasattr (OO0OOO0OO0O00O000 ,'bias_mask'):#line:12
        OO0OOO0OO0O00O000 .bias .data *=OO0OOO0OO0O00O000 .bias_mask #line:13
def mask_weight (O0OO0OOO0OO0O000O ,OO000O0000O0OO000 ,inplace =True ):#line:15
    ""#line:21
    if not inplace :#line:22
        O0OO0OOO0OO0O000O =deepcopy (O0OO0OOO0OO0O000O )#line:23
    if OO000O0000O0OO000 .shape !=O0OO0OOO0OO0O000O .weight .shape :#line:24
        return O0OO0OOO0OO0O000O #line:25
    OO000O0000O0OO000 =torch .tensor (OO000O0000O0OO000 ,dtype =O0OO0OOO0OO0O000O .weight .dtype ,device =O0OO0OOO0OO0O000O .weight .device ,requires_grad =False )#line:26
    if hasattr (O0OO0OOO0OO0O000O ,'weight_mask'):#line:27
        OO000O0000O0OO000 =OO000O0000O0OO000 +O0OO0OOO0OO0O000O .weight_mask #line:28
        OO000O0000O0OO000 [OO000O0000O0OO000 >0 ]=1 #line:29
        O0OO0OOO0OO0O000O .weight_mask =OO000O0000O0OO000 #line:30
    else :#line:31
        O0OO0OOO0OO0O000O .register_buffer ('weight_mask',OO000O0000O0OO000 )#line:32
    O0OO0OOO0OO0O000O .register_forward_pre_hook (_OO0OO00O0OOO00OO0 )#line:34
    return O0OO0OOO0OO0O000O #line:35
def mask_bias (OO000OOOO0O00OOOO ,O0000O0OO00O0O00O ,inplace =True ):#line:37
    ""#line:43
    if not inplace :#line:44
        OO000OOOO0O00OOOO =deepcopy (OO000OOOO0O00OOOO )#line:45
    if OO000OOOO0O00OOOO .bias is None or O0000O0OO00O0O00O .shape !=OO000OOOO0O00OOOO .bias .shape :#line:46
        return OO000OOOO0O00OOOO #line:47
    O0000O0OO00O0O00O =torch .tensor (O0000O0OO00O0O00O ,dtype =OO000OOOO0O00OOOO .weight .dtype ,device =OO000OOOO0O00OOOO .weight .device ,requires_grad =False )#line:49
    if hasattr (OO000OOOO0O00OOOO ,'bias_mask'):#line:50
        O0000O0OO00O0O00O =O0000O0OO00O0O00O +OO000OOOO0O00OOOO .bias_mask #line:51
        O0000O0OO00O0O00O [O0000O0OO00O0O00O >0 ]=1 #line:52
        OO000OOOO0O00OOOO .bias_mask =O0000O0OO00O0O00O #line:53
    else :#line:54
        OO000OOOO0O00OOOO .register_buffer ('bias_mask',O0000O0OO00O0O00O )#line:55
    OO000OOOO0O00OOOO .register_forward_pre_hook (_O0000OOOO00O000O0 )#line:56
    return OO000OOOO0O00OOOO #line:57
