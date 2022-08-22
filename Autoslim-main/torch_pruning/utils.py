from .dependency import TORCH_CONV ,TORCH_BATCHNORM ,TORCH_PRELU ,TORCH_LINEAR #line:1
def count_prunable_params (O0O00O000OO0OO0O0 ):#line:3
    if isinstance (O0O00O000OO0OO0O0 ,(TORCH_CONV ,TORCH_LINEAR )):#line:4
        OOO0OOO0OO0OO0000 =O0O00O000OO0OO0O0 .weight .numel ()#line:5
        if O0O00O000OO0OO0O0 .bias is not None :#line:6
            OOO0OOO0OO0OO0000 +=O0O00O000OO0OO0O0 .bias .numel ()#line:7
        return OOO0OOO0OO0OO0000 #line:8
    elif isinstance (O0O00O000OO0OO0O0 ,TORCH_BATCHNORM ):#line:9
        OOO0OOO0OO0OO0000 =O0O00O000OO0OO0O0 .running_mean .numel ()+O0O00O000OO0OO0O0 .running_var .numel ()#line:10
        if O0O00O000OO0OO0O0 .affine :#line:11
            OOO0OOO0OO0OO0000 +=O0O00O000OO0OO0O0 .weight .numel ()+O0O00O000OO0OO0O0 .bias .numel ()#line:12
        return OOO0OOO0OO0OO0000 #line:13
    elif isinstance (O0O00O000OO0OO0O0 ,TORCH_PRELU ):#line:14
        if len (O0O00O000OO0OO0O0 .weight )==1 :#line:15
            return 0 #line:16
        else :#line:17
            return O0O00O000OO0OO0O0 .weight .numel #line:18
    else :#line:19
        return 0 #line:20
def count_prunable_channels (O0OO0O00OO0OOO000 ):#line:22
    if isinstance (O0OO0O00OO0OOO000 ,TORCH_CONV ):#line:23
        return O0OO0O00OO0OOO000 .weight .shape [0 ]#line:24
    elif isinstance (O0OO0O00OO0OOO000 ,TORCH_LINEAR ):#line:25
        return O0OO0O00OO0OOO000 .out_features #line:26
    elif isinstance (O0OO0O00OO0OOO000 ,TORCH_BATCHNORM ):#line:27
        return O0OO0O00OO0OOO000 .num_features #line:28
    elif isinstance (O0OO0O00OO0OOO000 ,TORCH_PRELU ):#line:29
        if len (O0OO0O00OO0OOO000 .weight )==1 :#line:30
            return 0 #line:31
        else :#line:32
            return len (O0OO0O00OO0OOO000 .weight )#line:33
    else :#line:34
        return 0 #line:35
def count_params (O00OO0OOO00OO000O ):#line:37
    return sum ([O0O0O000OO0O00OOO .numel ()for O0O0O000OO0O00OOO in O00OO0OOO00OO000O .parameters ()])#line:38
