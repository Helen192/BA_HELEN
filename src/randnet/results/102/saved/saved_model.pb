��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758��	
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
x
dense_17029/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*!
shared_namedense_17029/bias
q
$dense_17029/bias/Read/ReadVariableOpReadVariableOpdense_17029/bias*
_output_shapes
:(*
dtype0
�
dense_17029/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*#
shared_namedense_17029/kernel
y
&dense_17029/kernel/Read/ReadVariableOpReadVariableOpdense_17029/kernel*
_output_shapes

:(*
dtype0
x
dense_17028/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namedense_17028/bias
q
$dense_17028/bias/Read/ReadVariableOpReadVariableOpdense_17028/bias*
_output_shapes
:*
dtype0
�
dense_17028/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*#
shared_namedense_17028/kernel
y
&dense_17028/kernel/Read/ReadVariableOpReadVariableOpdense_17028/kernel*
_output_shapes

:(*
dtype0
x
dense_17027/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*!
shared_namedense_17027/bias
q
$dense_17027/bias/Read/ReadVariableOpReadVariableOpdense_17027/bias*
_output_shapes
:(*
dtype0
�
dense_17027/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
(*#
shared_namedense_17027/kernel
y
&dense_17027/kernel/Read/ReadVariableOpReadVariableOpdense_17027/kernel*
_output_shapes

:
(*
dtype0
x
dense_17026/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_namedense_17026/bias
q
$dense_17026/bias/Read/ReadVariableOpReadVariableOpdense_17026/bias*
_output_shapes
:
*
dtype0
�
dense_17026/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(
*#
shared_namedense_17026/kernel
y
&dense_17026/kernel/Read/ReadVariableOpReadVariableOpdense_17026/kernel*
_output_shapes

:(
*
dtype0
x
dense_17025/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*!
shared_namedense_17025/bias
q
$dense_17025/bias/Read/ReadVariableOpReadVariableOpdense_17025/bias*
_output_shapes
:(*
dtype0
�
dense_17025/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*#
shared_namedense_17025/kernel
y
&dense_17025/kernel/Read/ReadVariableOpReadVariableOpdense_17025/kernel*
_output_shapes

:(*
dtype0
x
dense_17024/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namedense_17024/bias
q
$dense_17024/bias/Read/ReadVariableOpReadVariableOpdense_17024/bias*
_output_shapes
:*
dtype0
�
dense_17024/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*#
shared_namedense_17024/kernel
y
&dense_17024/kernel/Read/ReadVariableOpReadVariableOpdense_17024/kernel*
_output_shapes

:(*
dtype0
x
dense_17023/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*!
shared_namedense_17023/bias
q
$dense_17023/bias/Read/ReadVariableOpReadVariableOpdense_17023/bias*
_output_shapes
:(*
dtype0
�
dense_17023/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
(*#
shared_namedense_17023/kernel
y
&dense_17023/kernel/Read/ReadVariableOpReadVariableOpdense_17023/kernel*
_output_shapes

:
(*
dtype0
x
dense_17022/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_namedense_17022/bias
q
$dense_17022/bias/Read/ReadVariableOpReadVariableOpdense_17022/bias*
_output_shapes
:
*
dtype0
�
dense_17022/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(
*#
shared_namedense_17022/kernel
y
&dense_17022/kernel/Read/ReadVariableOpReadVariableOpdense_17022/kernel*
_output_shapes

:(
*
dtype0
x
dense_17021/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*!
shared_namedense_17021/bias
q
$dense_17021/bias/Read/ReadVariableOpReadVariableOpdense_17021/bias*
_output_shapes
:(*
dtype0
�
dense_17021/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*#
shared_namedense_17021/kernel
y
&dense_17021/kernel/Read/ReadVariableOpReadVariableOpdense_17021/kernel*
_output_shapes

:(*
dtype0
x
dense_17020/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namedense_17020/bias
q
$dense_17020/bias/Read/ReadVariableOpReadVariableOpdense_17020/bias*
_output_shapes
:*
dtype0
�
dense_17020/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*#
shared_namedense_17020/kernel
y
&dense_17020/kernel/Read/ReadVariableOpReadVariableOpdense_17020/kernel*
_output_shapes

:(*
dtype0
}
serving_default_input_3406Placeholder*'
_output_shapes
:���������(*
dtype0*
shape:���������(
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3406dense_17020/kerneldense_17020/biasdense_17021/kerneldense_17021/biasdense_17022/kerneldense_17022/biasdense_17023/kerneldense_17023/biasdense_17024/kerneldense_17024/biasdense_17025/kerneldense_17025/biasdense_17026/kerneldense_17026/biasdense_17027/kerneldense_17027/biasdense_17028/kerneldense_17028/biasdense_17029/kerneldense_17029/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� */
f*R(
&__inference_signature_wrapper_75416299

NoOpNoOp
�K
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�K
value�KB�K B�K
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
layer_with_weights-9
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias*
�
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

+kernel
,bias*
�
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

3kernel
4bias*
�
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

;kernel
<bias*
�
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses

Ckernel
Dbias*
�
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses

Kkernel
Lbias*
�
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses

Skernel
Tbias*
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses

[kernel
\bias*
�
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses

ckernel
dbias*
�
0
1
#2
$3
+4
,5
36
47
;8
<9
C10
D11
K12
L13
S14
T15
[16
\17
c18
d19*
�
0
1
#2
$3
+4
,5
36
47
;8
<9
C10
D11
K12
L13
S14
T15
[16
\17
c18
d19*
* 
�
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
jtrace_0
ktrace_1
ltrace_2
mtrace_3* 
6
ntrace_0
otrace_1
ptrace_2
qtrace_3* 
* 
O
r
_variables
s_iterations
t_learning_rate
u_update_step_xla*

vserving_default* 

0
1*

0
1*
* 
�
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

|trace_0* 

}trace_0* 
b\
VARIABLE_VALUEdense_17020/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEdense_17020/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

#0
$1*

#0
$1*
* 
�
~non_trainable_variables

layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
b\
VARIABLE_VALUEdense_17021/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEdense_17021/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

+0
,1*

+0
,1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
b\
VARIABLE_VALUEdense_17022/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEdense_17022/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

30
41*

30
41*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
b\
VARIABLE_VALUEdense_17023/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEdense_17023/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

;0
<1*

;0
<1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
b\
VARIABLE_VALUEdense_17024/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEdense_17024/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

C0
D1*

C0
D1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
b\
VARIABLE_VALUEdense_17025/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEdense_17025/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

K0
L1*

K0
L1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
b\
VARIABLE_VALUEdense_17026/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEdense_17026/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

S0
T1*

S0
T1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
b\
VARIABLE_VALUEdense_17027/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEdense_17027/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

[0
\1*

[0
\1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
b\
VARIABLE_VALUEdense_17028/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEdense_17028/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*

c0
d1*

c0
d1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
b\
VARIABLE_VALUEdense_17029/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEdense_17029/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
R
0
1
2
3
4
5
6
7
	8

9
10*

�0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

s0*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_17020/kerneldense_17020/biasdense_17021/kerneldense_17021/biasdense_17022/kerneldense_17022/biasdense_17023/kerneldense_17023/biasdense_17024/kerneldense_17024/biasdense_17025/kerneldense_17025/biasdense_17026/kerneldense_17026/biasdense_17027/kerneldense_17027/biasdense_17028/kerneldense_17028/biasdense_17029/kerneldense_17029/bias	iterationlearning_ratetotalcountConst*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_save_75416889
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_17020/kerneldense_17020/biasdense_17021/kerneldense_17021/biasdense_17022/kerneldense_17022/biasdense_17023/kerneldense_17023/biasdense_17024/kerneldense_17024/biasdense_17025/kerneldense_17025/biasdense_17026/kerneldense_17026/biasdense_17027/kerneldense_17027/biasdense_17028/kerneldense_17028/biasdense_17029/kerneldense_17029/bias	iterationlearning_ratetotalcount*$
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference__traced_restore_75416971��
�	
�
I__inference_dense_17029_layer_call_and_return_conditional_losses_75415800

inputs0
matmul_readvariableop_resource:(-
biasadd_readvariableop_resource:(
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
I__inference_dense_17024_layer_call_and_return_conditional_losses_75416625

inputs0
matmul_readvariableop_resource:(-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�

�
I__inference_dense_17028_layer_call_and_return_conditional_losses_75416703

inputs0
matmul_readvariableop_resource:(-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�	
�
I__inference_dense_17025_layer_call_and_return_conditional_losses_75415734

inputs0
matmul_readvariableop_resource:(-
biasadd_readvariableop_resource:(
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
I__inference_dense_17029_layer_call_and_return_conditional_losses_75416722

inputs0
matmul_readvariableop_resource:(-
biasadd_readvariableop_resource:(
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
.__inference_dense_17025_layer_call_fn_75416634

inputs
unknown:(
	unknown_0:(
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17025_layer_call_and_return_conditional_losses_75415734o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�7
�	
H__inference_model_3405_layer_call_and_return_conditional_losses_75415807

input_3406&
dense_17020_75415653:("
dense_17020_75415655:&
dense_17021_75415669:("
dense_17021_75415671:(&
dense_17022_75415686:(
"
dense_17022_75415688:
&
dense_17023_75415702:
("
dense_17023_75415704:(&
dense_17024_75415719:("
dense_17024_75415721:&
dense_17025_75415735:("
dense_17025_75415737:(&
dense_17026_75415752:(
"
dense_17026_75415754:
&
dense_17027_75415768:
("
dense_17027_75415770:(&
dense_17028_75415785:("
dense_17028_75415787:&
dense_17029_75415801:("
dense_17029_75415803:(
identity��#dense_17020/StatefulPartitionedCall�#dense_17021/StatefulPartitionedCall�#dense_17022/StatefulPartitionedCall�#dense_17023/StatefulPartitionedCall�#dense_17024/StatefulPartitionedCall�#dense_17025/StatefulPartitionedCall�#dense_17026/StatefulPartitionedCall�#dense_17027/StatefulPartitionedCall�#dense_17028/StatefulPartitionedCall�#dense_17029/StatefulPartitionedCall�
#dense_17020/StatefulPartitionedCallStatefulPartitionedCall
input_3406dense_17020_75415653dense_17020_75415655*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17020_layer_call_and_return_conditional_losses_75415652�
#dense_17021/StatefulPartitionedCallStatefulPartitionedCall,dense_17020/StatefulPartitionedCall:output:0dense_17021_75415669dense_17021_75415671*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17021_layer_call_and_return_conditional_losses_75415668�
#dense_17022/StatefulPartitionedCallStatefulPartitionedCall,dense_17021/StatefulPartitionedCall:output:0dense_17022_75415686dense_17022_75415688*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17022_layer_call_and_return_conditional_losses_75415685�
#dense_17023/StatefulPartitionedCallStatefulPartitionedCall,dense_17022/StatefulPartitionedCall:output:0dense_17023_75415702dense_17023_75415704*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17023_layer_call_and_return_conditional_losses_75415701�
#dense_17024/StatefulPartitionedCallStatefulPartitionedCall,dense_17023/StatefulPartitionedCall:output:0dense_17024_75415719dense_17024_75415721*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17024_layer_call_and_return_conditional_losses_75415718�
#dense_17025/StatefulPartitionedCallStatefulPartitionedCall,dense_17024/StatefulPartitionedCall:output:0dense_17025_75415735dense_17025_75415737*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17025_layer_call_and_return_conditional_losses_75415734�
#dense_17026/StatefulPartitionedCallStatefulPartitionedCall,dense_17025/StatefulPartitionedCall:output:0dense_17026_75415752dense_17026_75415754*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17026_layer_call_and_return_conditional_losses_75415751�
#dense_17027/StatefulPartitionedCallStatefulPartitionedCall,dense_17026/StatefulPartitionedCall:output:0dense_17027_75415768dense_17027_75415770*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17027_layer_call_and_return_conditional_losses_75415767�
#dense_17028/StatefulPartitionedCallStatefulPartitionedCall,dense_17027/StatefulPartitionedCall:output:0dense_17028_75415785dense_17028_75415787*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17028_layer_call_and_return_conditional_losses_75415784�
#dense_17029/StatefulPartitionedCallStatefulPartitionedCall,dense_17028/StatefulPartitionedCall:output:0dense_17029_75415801dense_17029_75415803*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17029_layer_call_and_return_conditional_losses_75415800{
IdentityIdentity,dense_17029/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(�
NoOpNoOp$^dense_17020/StatefulPartitionedCall$^dense_17021/StatefulPartitionedCall$^dense_17022/StatefulPartitionedCall$^dense_17023/StatefulPartitionedCall$^dense_17024/StatefulPartitionedCall$^dense_17025/StatefulPartitionedCall$^dense_17026/StatefulPartitionedCall$^dense_17027/StatefulPartitionedCall$^dense_17028/StatefulPartitionedCall$^dense_17029/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������(: : : : : : : : : : : : : : : : : : : : 2J
#dense_17020/StatefulPartitionedCall#dense_17020/StatefulPartitionedCall2J
#dense_17021/StatefulPartitionedCall#dense_17021/StatefulPartitionedCall2J
#dense_17022/StatefulPartitionedCall#dense_17022/StatefulPartitionedCall2J
#dense_17023/StatefulPartitionedCall#dense_17023/StatefulPartitionedCall2J
#dense_17024/StatefulPartitionedCall#dense_17024/StatefulPartitionedCall2J
#dense_17025/StatefulPartitionedCall#dense_17025/StatefulPartitionedCall2J
#dense_17026/StatefulPartitionedCall#dense_17026/StatefulPartitionedCall2J
#dense_17027/StatefulPartitionedCall#dense_17027/StatefulPartitionedCall2J
#dense_17028/StatefulPartitionedCall#dense_17028/StatefulPartitionedCall2J
#dense_17029/StatefulPartitionedCall#dense_17029/StatefulPartitionedCall:S O
'
_output_shapes
:���������(
$
_user_specified_name
input_3406
�

�
I__inference_dense_17022_layer_call_and_return_conditional_losses_75416586

inputs0
matmul_readvariableop_resource:(
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�

�
I__inference_dense_17022_layer_call_and_return_conditional_losses_75415685

inputs0
matmul_readvariableop_resource:(
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�
�
.__inference_dense_17024_layer_call_fn_75416614

inputs
unknown:(
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17024_layer_call_and_return_conditional_losses_75415718o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������(: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�	
�
I__inference_dense_17021_layer_call_and_return_conditional_losses_75416566

inputs0
matmul_readvariableop_resource:(-
biasadd_readvariableop_resource:(
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
I__inference_dense_17020_layer_call_and_return_conditional_losses_75415652

inputs0
matmul_readvariableop_resource:(-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�h
�
#__inference__wrapped_model_75415637

input_3406G
5model_3405_dense_17020_matmul_readvariableop_resource:(D
6model_3405_dense_17020_biasadd_readvariableop_resource:G
5model_3405_dense_17021_matmul_readvariableop_resource:(D
6model_3405_dense_17021_biasadd_readvariableop_resource:(G
5model_3405_dense_17022_matmul_readvariableop_resource:(
D
6model_3405_dense_17022_biasadd_readvariableop_resource:
G
5model_3405_dense_17023_matmul_readvariableop_resource:
(D
6model_3405_dense_17023_biasadd_readvariableop_resource:(G
5model_3405_dense_17024_matmul_readvariableop_resource:(D
6model_3405_dense_17024_biasadd_readvariableop_resource:G
5model_3405_dense_17025_matmul_readvariableop_resource:(D
6model_3405_dense_17025_biasadd_readvariableop_resource:(G
5model_3405_dense_17026_matmul_readvariableop_resource:(
D
6model_3405_dense_17026_biasadd_readvariableop_resource:
G
5model_3405_dense_17027_matmul_readvariableop_resource:
(D
6model_3405_dense_17027_biasadd_readvariableop_resource:(G
5model_3405_dense_17028_matmul_readvariableop_resource:(D
6model_3405_dense_17028_biasadd_readvariableop_resource:G
5model_3405_dense_17029_matmul_readvariableop_resource:(D
6model_3405_dense_17029_biasadd_readvariableop_resource:(
identity��-model_3405/dense_17020/BiasAdd/ReadVariableOp�,model_3405/dense_17020/MatMul/ReadVariableOp�-model_3405/dense_17021/BiasAdd/ReadVariableOp�,model_3405/dense_17021/MatMul/ReadVariableOp�-model_3405/dense_17022/BiasAdd/ReadVariableOp�,model_3405/dense_17022/MatMul/ReadVariableOp�-model_3405/dense_17023/BiasAdd/ReadVariableOp�,model_3405/dense_17023/MatMul/ReadVariableOp�-model_3405/dense_17024/BiasAdd/ReadVariableOp�,model_3405/dense_17024/MatMul/ReadVariableOp�-model_3405/dense_17025/BiasAdd/ReadVariableOp�,model_3405/dense_17025/MatMul/ReadVariableOp�-model_3405/dense_17026/BiasAdd/ReadVariableOp�,model_3405/dense_17026/MatMul/ReadVariableOp�-model_3405/dense_17027/BiasAdd/ReadVariableOp�,model_3405/dense_17027/MatMul/ReadVariableOp�-model_3405/dense_17028/BiasAdd/ReadVariableOp�,model_3405/dense_17028/MatMul/ReadVariableOp�-model_3405/dense_17029/BiasAdd/ReadVariableOp�,model_3405/dense_17029/MatMul/ReadVariableOp�
,model_3405/dense_17020/MatMul/ReadVariableOpReadVariableOp5model_3405_dense_17020_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
model_3405/dense_17020/MatMulMatMul
input_34064model_3405/dense_17020/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-model_3405/dense_17020/BiasAdd/ReadVariableOpReadVariableOp6model_3405_dense_17020_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_3405/dense_17020/BiasAddBiasAdd'model_3405/dense_17020/MatMul:product:05model_3405/dense_17020/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
model_3405/dense_17020/ReluRelu'model_3405/dense_17020/BiasAdd:output:0*
T0*'
_output_shapes
:����������
,model_3405/dense_17021/MatMul/ReadVariableOpReadVariableOp5model_3405_dense_17021_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
model_3405/dense_17021/MatMulMatMul)model_3405/dense_17020/Relu:activations:04model_3405/dense_17021/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
-model_3405/dense_17021/BiasAdd/ReadVariableOpReadVariableOp6model_3405_dense_17021_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
model_3405/dense_17021/BiasAddBiasAdd'model_3405/dense_17021/MatMul:product:05model_3405/dense_17021/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
,model_3405/dense_17022/MatMul/ReadVariableOpReadVariableOp5model_3405_dense_17022_matmul_readvariableop_resource*
_output_shapes

:(
*
dtype0�
model_3405/dense_17022/MatMulMatMul'model_3405/dense_17021/BiasAdd:output:04model_3405/dense_17022/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
-model_3405/dense_17022/BiasAdd/ReadVariableOpReadVariableOp6model_3405_dense_17022_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
model_3405/dense_17022/BiasAddBiasAdd'model_3405/dense_17022/MatMul:product:05model_3405/dense_17022/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
~
model_3405/dense_17022/ReluRelu'model_3405/dense_17022/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
,model_3405/dense_17023/MatMul/ReadVariableOpReadVariableOp5model_3405_dense_17023_matmul_readvariableop_resource*
_output_shapes

:
(*
dtype0�
model_3405/dense_17023/MatMulMatMul)model_3405/dense_17022/Relu:activations:04model_3405/dense_17023/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
-model_3405/dense_17023/BiasAdd/ReadVariableOpReadVariableOp6model_3405_dense_17023_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
model_3405/dense_17023/BiasAddBiasAdd'model_3405/dense_17023/MatMul:product:05model_3405/dense_17023/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
,model_3405/dense_17024/MatMul/ReadVariableOpReadVariableOp5model_3405_dense_17024_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
model_3405/dense_17024/MatMulMatMul'model_3405/dense_17023/BiasAdd:output:04model_3405/dense_17024/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-model_3405/dense_17024/BiasAdd/ReadVariableOpReadVariableOp6model_3405_dense_17024_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_3405/dense_17024/BiasAddBiasAdd'model_3405/dense_17024/MatMul:product:05model_3405/dense_17024/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
model_3405/dense_17024/ReluRelu'model_3405/dense_17024/BiasAdd:output:0*
T0*'
_output_shapes
:����������
,model_3405/dense_17025/MatMul/ReadVariableOpReadVariableOp5model_3405_dense_17025_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
model_3405/dense_17025/MatMulMatMul)model_3405/dense_17024/Relu:activations:04model_3405/dense_17025/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
-model_3405/dense_17025/BiasAdd/ReadVariableOpReadVariableOp6model_3405_dense_17025_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
model_3405/dense_17025/BiasAddBiasAdd'model_3405/dense_17025/MatMul:product:05model_3405/dense_17025/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
,model_3405/dense_17026/MatMul/ReadVariableOpReadVariableOp5model_3405_dense_17026_matmul_readvariableop_resource*
_output_shapes

:(
*
dtype0�
model_3405/dense_17026/MatMulMatMul'model_3405/dense_17025/BiasAdd:output:04model_3405/dense_17026/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
-model_3405/dense_17026/BiasAdd/ReadVariableOpReadVariableOp6model_3405_dense_17026_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
model_3405/dense_17026/BiasAddBiasAdd'model_3405/dense_17026/MatMul:product:05model_3405/dense_17026/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
~
model_3405/dense_17026/ReluRelu'model_3405/dense_17026/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
,model_3405/dense_17027/MatMul/ReadVariableOpReadVariableOp5model_3405_dense_17027_matmul_readvariableop_resource*
_output_shapes

:
(*
dtype0�
model_3405/dense_17027/MatMulMatMul)model_3405/dense_17026/Relu:activations:04model_3405/dense_17027/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
-model_3405/dense_17027/BiasAdd/ReadVariableOpReadVariableOp6model_3405_dense_17027_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
model_3405/dense_17027/BiasAddBiasAdd'model_3405/dense_17027/MatMul:product:05model_3405/dense_17027/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
,model_3405/dense_17028/MatMul/ReadVariableOpReadVariableOp5model_3405_dense_17028_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
model_3405/dense_17028/MatMulMatMul'model_3405/dense_17027/BiasAdd:output:04model_3405/dense_17028/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-model_3405/dense_17028/BiasAdd/ReadVariableOpReadVariableOp6model_3405_dense_17028_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_3405/dense_17028/BiasAddBiasAdd'model_3405/dense_17028/MatMul:product:05model_3405/dense_17028/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
model_3405/dense_17028/ReluRelu'model_3405/dense_17028/BiasAdd:output:0*
T0*'
_output_shapes
:����������
,model_3405/dense_17029/MatMul/ReadVariableOpReadVariableOp5model_3405_dense_17029_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
model_3405/dense_17029/MatMulMatMul)model_3405/dense_17028/Relu:activations:04model_3405/dense_17029/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
-model_3405/dense_17029/BiasAdd/ReadVariableOpReadVariableOp6model_3405_dense_17029_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
model_3405/dense_17029/BiasAddBiasAdd'model_3405/dense_17029/MatMul:product:05model_3405/dense_17029/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(v
IdentityIdentity'model_3405/dense_17029/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������(�
NoOpNoOp.^model_3405/dense_17020/BiasAdd/ReadVariableOp-^model_3405/dense_17020/MatMul/ReadVariableOp.^model_3405/dense_17021/BiasAdd/ReadVariableOp-^model_3405/dense_17021/MatMul/ReadVariableOp.^model_3405/dense_17022/BiasAdd/ReadVariableOp-^model_3405/dense_17022/MatMul/ReadVariableOp.^model_3405/dense_17023/BiasAdd/ReadVariableOp-^model_3405/dense_17023/MatMul/ReadVariableOp.^model_3405/dense_17024/BiasAdd/ReadVariableOp-^model_3405/dense_17024/MatMul/ReadVariableOp.^model_3405/dense_17025/BiasAdd/ReadVariableOp-^model_3405/dense_17025/MatMul/ReadVariableOp.^model_3405/dense_17026/BiasAdd/ReadVariableOp-^model_3405/dense_17026/MatMul/ReadVariableOp.^model_3405/dense_17027/BiasAdd/ReadVariableOp-^model_3405/dense_17027/MatMul/ReadVariableOp.^model_3405/dense_17028/BiasAdd/ReadVariableOp-^model_3405/dense_17028/MatMul/ReadVariableOp.^model_3405/dense_17029/BiasAdd/ReadVariableOp-^model_3405/dense_17029/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������(: : : : : : : : : : : : : : : : : : : : 2^
-model_3405/dense_17020/BiasAdd/ReadVariableOp-model_3405/dense_17020/BiasAdd/ReadVariableOp2\
,model_3405/dense_17020/MatMul/ReadVariableOp,model_3405/dense_17020/MatMul/ReadVariableOp2^
-model_3405/dense_17021/BiasAdd/ReadVariableOp-model_3405/dense_17021/BiasAdd/ReadVariableOp2\
,model_3405/dense_17021/MatMul/ReadVariableOp,model_3405/dense_17021/MatMul/ReadVariableOp2^
-model_3405/dense_17022/BiasAdd/ReadVariableOp-model_3405/dense_17022/BiasAdd/ReadVariableOp2\
,model_3405/dense_17022/MatMul/ReadVariableOp,model_3405/dense_17022/MatMul/ReadVariableOp2^
-model_3405/dense_17023/BiasAdd/ReadVariableOp-model_3405/dense_17023/BiasAdd/ReadVariableOp2\
,model_3405/dense_17023/MatMul/ReadVariableOp,model_3405/dense_17023/MatMul/ReadVariableOp2^
-model_3405/dense_17024/BiasAdd/ReadVariableOp-model_3405/dense_17024/BiasAdd/ReadVariableOp2\
,model_3405/dense_17024/MatMul/ReadVariableOp,model_3405/dense_17024/MatMul/ReadVariableOp2^
-model_3405/dense_17025/BiasAdd/ReadVariableOp-model_3405/dense_17025/BiasAdd/ReadVariableOp2\
,model_3405/dense_17025/MatMul/ReadVariableOp,model_3405/dense_17025/MatMul/ReadVariableOp2^
-model_3405/dense_17026/BiasAdd/ReadVariableOp-model_3405/dense_17026/BiasAdd/ReadVariableOp2\
,model_3405/dense_17026/MatMul/ReadVariableOp,model_3405/dense_17026/MatMul/ReadVariableOp2^
-model_3405/dense_17027/BiasAdd/ReadVariableOp-model_3405/dense_17027/BiasAdd/ReadVariableOp2\
,model_3405/dense_17027/MatMul/ReadVariableOp,model_3405/dense_17027/MatMul/ReadVariableOp2^
-model_3405/dense_17028/BiasAdd/ReadVariableOp-model_3405/dense_17028/BiasAdd/ReadVariableOp2\
,model_3405/dense_17028/MatMul/ReadVariableOp,model_3405/dense_17028/MatMul/ReadVariableOp2^
-model_3405/dense_17029/BiasAdd/ReadVariableOp-model_3405/dense_17029/BiasAdd/ReadVariableOp2\
,model_3405/dense_17029/MatMul/ReadVariableOp,model_3405/dense_17029/MatMul/ReadVariableOp:S O
'
_output_shapes
:���������(
$
_user_specified_name
input_3406
�	
�
I__inference_dense_17023_layer_call_and_return_conditional_losses_75416605

inputs0
matmul_readvariableop_resource:
(-
biasadd_readvariableop_resource:(
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
.__inference_dense_17021_layer_call_fn_75416556

inputs
unknown:(
	unknown_0:(
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17021_layer_call_and_return_conditional_losses_75415668o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
.__inference_dense_17022_layer_call_fn_75416575

inputs
unknown:(

	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17022_layer_call_and_return_conditional_losses_75415685o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������(: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�7
�	
H__inference_model_3405_layer_call_and_return_conditional_losses_75416017

inputs&
dense_17020_75415966:("
dense_17020_75415968:&
dense_17021_75415971:("
dense_17021_75415973:(&
dense_17022_75415976:(
"
dense_17022_75415978:
&
dense_17023_75415981:
("
dense_17023_75415983:(&
dense_17024_75415986:("
dense_17024_75415988:&
dense_17025_75415991:("
dense_17025_75415993:(&
dense_17026_75415996:(
"
dense_17026_75415998:
&
dense_17027_75416001:
("
dense_17027_75416003:(&
dense_17028_75416006:("
dense_17028_75416008:&
dense_17029_75416011:("
dense_17029_75416013:(
identity��#dense_17020/StatefulPartitionedCall�#dense_17021/StatefulPartitionedCall�#dense_17022/StatefulPartitionedCall�#dense_17023/StatefulPartitionedCall�#dense_17024/StatefulPartitionedCall�#dense_17025/StatefulPartitionedCall�#dense_17026/StatefulPartitionedCall�#dense_17027/StatefulPartitionedCall�#dense_17028/StatefulPartitionedCall�#dense_17029/StatefulPartitionedCall�
#dense_17020/StatefulPartitionedCallStatefulPartitionedCallinputsdense_17020_75415966dense_17020_75415968*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17020_layer_call_and_return_conditional_losses_75415652�
#dense_17021/StatefulPartitionedCallStatefulPartitionedCall,dense_17020/StatefulPartitionedCall:output:0dense_17021_75415971dense_17021_75415973*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17021_layer_call_and_return_conditional_losses_75415668�
#dense_17022/StatefulPartitionedCallStatefulPartitionedCall,dense_17021/StatefulPartitionedCall:output:0dense_17022_75415976dense_17022_75415978*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17022_layer_call_and_return_conditional_losses_75415685�
#dense_17023/StatefulPartitionedCallStatefulPartitionedCall,dense_17022/StatefulPartitionedCall:output:0dense_17023_75415981dense_17023_75415983*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17023_layer_call_and_return_conditional_losses_75415701�
#dense_17024/StatefulPartitionedCallStatefulPartitionedCall,dense_17023/StatefulPartitionedCall:output:0dense_17024_75415986dense_17024_75415988*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17024_layer_call_and_return_conditional_losses_75415718�
#dense_17025/StatefulPartitionedCallStatefulPartitionedCall,dense_17024/StatefulPartitionedCall:output:0dense_17025_75415991dense_17025_75415993*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17025_layer_call_and_return_conditional_losses_75415734�
#dense_17026/StatefulPartitionedCallStatefulPartitionedCall,dense_17025/StatefulPartitionedCall:output:0dense_17026_75415996dense_17026_75415998*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17026_layer_call_and_return_conditional_losses_75415751�
#dense_17027/StatefulPartitionedCallStatefulPartitionedCall,dense_17026/StatefulPartitionedCall:output:0dense_17027_75416001dense_17027_75416003*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17027_layer_call_and_return_conditional_losses_75415767�
#dense_17028/StatefulPartitionedCallStatefulPartitionedCall,dense_17027/StatefulPartitionedCall:output:0dense_17028_75416006dense_17028_75416008*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17028_layer_call_and_return_conditional_losses_75415784�
#dense_17029/StatefulPartitionedCallStatefulPartitionedCall,dense_17028/StatefulPartitionedCall:output:0dense_17029_75416011dense_17029_75416013*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17029_layer_call_and_return_conditional_losses_75415800{
IdentityIdentity,dense_17029/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(�
NoOpNoOp$^dense_17020/StatefulPartitionedCall$^dense_17021/StatefulPartitionedCall$^dense_17022/StatefulPartitionedCall$^dense_17023/StatefulPartitionedCall$^dense_17024/StatefulPartitionedCall$^dense_17025/StatefulPartitionedCall$^dense_17026/StatefulPartitionedCall$^dense_17027/StatefulPartitionedCall$^dense_17028/StatefulPartitionedCall$^dense_17029/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������(: : : : : : : : : : : : : : : : : : : : 2J
#dense_17020/StatefulPartitionedCall#dense_17020/StatefulPartitionedCall2J
#dense_17021/StatefulPartitionedCall#dense_17021/StatefulPartitionedCall2J
#dense_17022/StatefulPartitionedCall#dense_17022/StatefulPartitionedCall2J
#dense_17023/StatefulPartitionedCall#dense_17023/StatefulPartitionedCall2J
#dense_17024/StatefulPartitionedCall#dense_17024/StatefulPartitionedCall2J
#dense_17025/StatefulPartitionedCall#dense_17025/StatefulPartitionedCall2J
#dense_17026/StatefulPartitionedCall#dense_17026/StatefulPartitionedCall2J
#dense_17027/StatefulPartitionedCall#dense_17027/StatefulPartitionedCall2J
#dense_17028/StatefulPartitionedCall#dense_17028/StatefulPartitionedCall2J
#dense_17029/StatefulPartitionedCall#dense_17029/StatefulPartitionedCall:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�	
�
I__inference_dense_17021_layer_call_and_return_conditional_losses_75415668

inputs0
matmul_readvariableop_resource:(-
biasadd_readvariableop_resource:(
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
.__inference_dense_17023_layer_call_fn_75416595

inputs
unknown:
(
	unknown_0:(
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17023_layer_call_and_return_conditional_losses_75415701o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
-__inference_model_3405_layer_call_fn_75415961

input_3406
unknown:(
	unknown_0:
	unknown_1:(
	unknown_2:(
	unknown_3:(

	unknown_4:

	unknown_5:
(
	unknown_6:(
	unknown_7:(
	unknown_8:
	unknown_9:(

unknown_10:(

unknown_11:(


unknown_12:


unknown_13:
(

unknown_14:(

unknown_15:(

unknown_16:

unknown_17:(

unknown_18:(
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
input_3406unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_model_3405_layer_call_and_return_conditional_losses_75415918o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������(: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:���������(
$
_user_specified_name
input_3406
�	
�
I__inference_dense_17027_layer_call_and_return_conditional_losses_75416683

inputs0
matmul_readvariableop_resource:
(-
biasadd_readvariableop_resource:(
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
.__inference_dense_17026_layer_call_fn_75416653

inputs
unknown:(

	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17026_layer_call_and_return_conditional_losses_75415751o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������(: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�	
�
I__inference_dense_17023_layer_call_and_return_conditional_losses_75415701

inputs0
matmul_readvariableop_resource:
(-
biasadd_readvariableop_resource:(
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
-__inference_model_3405_layer_call_fn_75416389

inputs
unknown:(
	unknown_0:
	unknown_1:(
	unknown_2:(
	unknown_3:(

	unknown_4:

	unknown_5:
(
	unknown_6:(
	unknown_7:(
	unknown_8:
	unknown_9:(

unknown_10:(

unknown_11:(


unknown_12:


unknown_13:
(

unknown_14:(

unknown_15:(

unknown_16:

unknown_17:(

unknown_18:(
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_model_3405_layer_call_and_return_conditional_losses_75416017o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������(: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�

�
I__inference_dense_17026_layer_call_and_return_conditional_losses_75416664

inputs0
matmul_readvariableop_resource:(
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�	
�
I__inference_dense_17025_layer_call_and_return_conditional_losses_75416644

inputs0
matmul_readvariableop_resource:(-
biasadd_readvariableop_resource:(
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�V
�
H__inference_model_3405_layer_call_and_return_conditional_losses_75416458

inputs<
*dense_17020_matmul_readvariableop_resource:(9
+dense_17020_biasadd_readvariableop_resource:<
*dense_17021_matmul_readvariableop_resource:(9
+dense_17021_biasadd_readvariableop_resource:(<
*dense_17022_matmul_readvariableop_resource:(
9
+dense_17022_biasadd_readvariableop_resource:
<
*dense_17023_matmul_readvariableop_resource:
(9
+dense_17023_biasadd_readvariableop_resource:(<
*dense_17024_matmul_readvariableop_resource:(9
+dense_17024_biasadd_readvariableop_resource:<
*dense_17025_matmul_readvariableop_resource:(9
+dense_17025_biasadd_readvariableop_resource:(<
*dense_17026_matmul_readvariableop_resource:(
9
+dense_17026_biasadd_readvariableop_resource:
<
*dense_17027_matmul_readvariableop_resource:
(9
+dense_17027_biasadd_readvariableop_resource:(<
*dense_17028_matmul_readvariableop_resource:(9
+dense_17028_biasadd_readvariableop_resource:<
*dense_17029_matmul_readvariableop_resource:(9
+dense_17029_biasadd_readvariableop_resource:(
identity��"dense_17020/BiasAdd/ReadVariableOp�!dense_17020/MatMul/ReadVariableOp�"dense_17021/BiasAdd/ReadVariableOp�!dense_17021/MatMul/ReadVariableOp�"dense_17022/BiasAdd/ReadVariableOp�!dense_17022/MatMul/ReadVariableOp�"dense_17023/BiasAdd/ReadVariableOp�!dense_17023/MatMul/ReadVariableOp�"dense_17024/BiasAdd/ReadVariableOp�!dense_17024/MatMul/ReadVariableOp�"dense_17025/BiasAdd/ReadVariableOp�!dense_17025/MatMul/ReadVariableOp�"dense_17026/BiasAdd/ReadVariableOp�!dense_17026/MatMul/ReadVariableOp�"dense_17027/BiasAdd/ReadVariableOp�!dense_17027/MatMul/ReadVariableOp�"dense_17028/BiasAdd/ReadVariableOp�!dense_17028/MatMul/ReadVariableOp�"dense_17029/BiasAdd/ReadVariableOp�!dense_17029/MatMul/ReadVariableOp�
!dense_17020/MatMul/ReadVariableOpReadVariableOp*dense_17020_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
dense_17020/MatMulMatMulinputs)dense_17020/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"dense_17020/BiasAdd/ReadVariableOpReadVariableOp+dense_17020_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_17020/BiasAddBiasAdddense_17020/MatMul:product:0*dense_17020/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
dense_17020/ReluReludense_17020/BiasAdd:output:0*
T0*'
_output_shapes
:����������
!dense_17021/MatMul/ReadVariableOpReadVariableOp*dense_17021_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
dense_17021/MatMulMatMuldense_17020/Relu:activations:0)dense_17021/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
"dense_17021/BiasAdd/ReadVariableOpReadVariableOp+dense_17021_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
dense_17021/BiasAddBiasAdddense_17021/MatMul:product:0*dense_17021/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
!dense_17022/MatMul/ReadVariableOpReadVariableOp*dense_17022_matmul_readvariableop_resource*
_output_shapes

:(
*
dtype0�
dense_17022/MatMulMatMuldense_17021/BiasAdd:output:0)dense_17022/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
"dense_17022/BiasAdd/ReadVariableOpReadVariableOp+dense_17022_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
dense_17022/BiasAddBiasAdddense_17022/MatMul:product:0*dense_17022/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
h
dense_17022/ReluReludense_17022/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
!dense_17023/MatMul/ReadVariableOpReadVariableOp*dense_17023_matmul_readvariableop_resource*
_output_shapes

:
(*
dtype0�
dense_17023/MatMulMatMuldense_17022/Relu:activations:0)dense_17023/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
"dense_17023/BiasAdd/ReadVariableOpReadVariableOp+dense_17023_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
dense_17023/BiasAddBiasAdddense_17023/MatMul:product:0*dense_17023/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
!dense_17024/MatMul/ReadVariableOpReadVariableOp*dense_17024_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
dense_17024/MatMulMatMuldense_17023/BiasAdd:output:0)dense_17024/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"dense_17024/BiasAdd/ReadVariableOpReadVariableOp+dense_17024_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_17024/BiasAddBiasAdddense_17024/MatMul:product:0*dense_17024/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
dense_17024/ReluReludense_17024/BiasAdd:output:0*
T0*'
_output_shapes
:����������
!dense_17025/MatMul/ReadVariableOpReadVariableOp*dense_17025_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
dense_17025/MatMulMatMuldense_17024/Relu:activations:0)dense_17025/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
"dense_17025/BiasAdd/ReadVariableOpReadVariableOp+dense_17025_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
dense_17025/BiasAddBiasAdddense_17025/MatMul:product:0*dense_17025/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
!dense_17026/MatMul/ReadVariableOpReadVariableOp*dense_17026_matmul_readvariableop_resource*
_output_shapes

:(
*
dtype0�
dense_17026/MatMulMatMuldense_17025/BiasAdd:output:0)dense_17026/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
"dense_17026/BiasAdd/ReadVariableOpReadVariableOp+dense_17026_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
dense_17026/BiasAddBiasAdddense_17026/MatMul:product:0*dense_17026/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
h
dense_17026/ReluReludense_17026/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
!dense_17027/MatMul/ReadVariableOpReadVariableOp*dense_17027_matmul_readvariableop_resource*
_output_shapes

:
(*
dtype0�
dense_17027/MatMulMatMuldense_17026/Relu:activations:0)dense_17027/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
"dense_17027/BiasAdd/ReadVariableOpReadVariableOp+dense_17027_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
dense_17027/BiasAddBiasAdddense_17027/MatMul:product:0*dense_17027/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
!dense_17028/MatMul/ReadVariableOpReadVariableOp*dense_17028_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
dense_17028/MatMulMatMuldense_17027/BiasAdd:output:0)dense_17028/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"dense_17028/BiasAdd/ReadVariableOpReadVariableOp+dense_17028_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_17028/BiasAddBiasAdddense_17028/MatMul:product:0*dense_17028/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
dense_17028/ReluReludense_17028/BiasAdd:output:0*
T0*'
_output_shapes
:����������
!dense_17029/MatMul/ReadVariableOpReadVariableOp*dense_17029_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
dense_17029/MatMulMatMuldense_17028/Relu:activations:0)dense_17029/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
"dense_17029/BiasAdd/ReadVariableOpReadVariableOp+dense_17029_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
dense_17029/BiasAddBiasAdddense_17029/MatMul:product:0*dense_17029/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(k
IdentityIdentitydense_17029/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������(�
NoOpNoOp#^dense_17020/BiasAdd/ReadVariableOp"^dense_17020/MatMul/ReadVariableOp#^dense_17021/BiasAdd/ReadVariableOp"^dense_17021/MatMul/ReadVariableOp#^dense_17022/BiasAdd/ReadVariableOp"^dense_17022/MatMul/ReadVariableOp#^dense_17023/BiasAdd/ReadVariableOp"^dense_17023/MatMul/ReadVariableOp#^dense_17024/BiasAdd/ReadVariableOp"^dense_17024/MatMul/ReadVariableOp#^dense_17025/BiasAdd/ReadVariableOp"^dense_17025/MatMul/ReadVariableOp#^dense_17026/BiasAdd/ReadVariableOp"^dense_17026/MatMul/ReadVariableOp#^dense_17027/BiasAdd/ReadVariableOp"^dense_17027/MatMul/ReadVariableOp#^dense_17028/BiasAdd/ReadVariableOp"^dense_17028/MatMul/ReadVariableOp#^dense_17029/BiasAdd/ReadVariableOp"^dense_17029/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������(: : : : : : : : : : : : : : : : : : : : 2H
"dense_17020/BiasAdd/ReadVariableOp"dense_17020/BiasAdd/ReadVariableOp2F
!dense_17020/MatMul/ReadVariableOp!dense_17020/MatMul/ReadVariableOp2H
"dense_17021/BiasAdd/ReadVariableOp"dense_17021/BiasAdd/ReadVariableOp2F
!dense_17021/MatMul/ReadVariableOp!dense_17021/MatMul/ReadVariableOp2H
"dense_17022/BiasAdd/ReadVariableOp"dense_17022/BiasAdd/ReadVariableOp2F
!dense_17022/MatMul/ReadVariableOp!dense_17022/MatMul/ReadVariableOp2H
"dense_17023/BiasAdd/ReadVariableOp"dense_17023/BiasAdd/ReadVariableOp2F
!dense_17023/MatMul/ReadVariableOp!dense_17023/MatMul/ReadVariableOp2H
"dense_17024/BiasAdd/ReadVariableOp"dense_17024/BiasAdd/ReadVariableOp2F
!dense_17024/MatMul/ReadVariableOp!dense_17024/MatMul/ReadVariableOp2H
"dense_17025/BiasAdd/ReadVariableOp"dense_17025/BiasAdd/ReadVariableOp2F
!dense_17025/MatMul/ReadVariableOp!dense_17025/MatMul/ReadVariableOp2H
"dense_17026/BiasAdd/ReadVariableOp"dense_17026/BiasAdd/ReadVariableOp2F
!dense_17026/MatMul/ReadVariableOp!dense_17026/MatMul/ReadVariableOp2H
"dense_17027/BiasAdd/ReadVariableOp"dense_17027/BiasAdd/ReadVariableOp2F
!dense_17027/MatMul/ReadVariableOp!dense_17027/MatMul/ReadVariableOp2H
"dense_17028/BiasAdd/ReadVariableOp"dense_17028/BiasAdd/ReadVariableOp2F
!dense_17028/MatMul/ReadVariableOp!dense_17028/MatMul/ReadVariableOp2H
"dense_17029/BiasAdd/ReadVariableOp"dense_17029/BiasAdd/ReadVariableOp2F
!dense_17029/MatMul/ReadVariableOp!dense_17029/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�7
�	
H__inference_model_3405_layer_call_and_return_conditional_losses_75415918

inputs&
dense_17020_75415867:("
dense_17020_75415869:&
dense_17021_75415872:("
dense_17021_75415874:(&
dense_17022_75415877:(
"
dense_17022_75415879:
&
dense_17023_75415882:
("
dense_17023_75415884:(&
dense_17024_75415887:("
dense_17024_75415889:&
dense_17025_75415892:("
dense_17025_75415894:(&
dense_17026_75415897:(
"
dense_17026_75415899:
&
dense_17027_75415902:
("
dense_17027_75415904:(&
dense_17028_75415907:("
dense_17028_75415909:&
dense_17029_75415912:("
dense_17029_75415914:(
identity��#dense_17020/StatefulPartitionedCall�#dense_17021/StatefulPartitionedCall�#dense_17022/StatefulPartitionedCall�#dense_17023/StatefulPartitionedCall�#dense_17024/StatefulPartitionedCall�#dense_17025/StatefulPartitionedCall�#dense_17026/StatefulPartitionedCall�#dense_17027/StatefulPartitionedCall�#dense_17028/StatefulPartitionedCall�#dense_17029/StatefulPartitionedCall�
#dense_17020/StatefulPartitionedCallStatefulPartitionedCallinputsdense_17020_75415867dense_17020_75415869*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17020_layer_call_and_return_conditional_losses_75415652�
#dense_17021/StatefulPartitionedCallStatefulPartitionedCall,dense_17020/StatefulPartitionedCall:output:0dense_17021_75415872dense_17021_75415874*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17021_layer_call_and_return_conditional_losses_75415668�
#dense_17022/StatefulPartitionedCallStatefulPartitionedCall,dense_17021/StatefulPartitionedCall:output:0dense_17022_75415877dense_17022_75415879*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17022_layer_call_and_return_conditional_losses_75415685�
#dense_17023/StatefulPartitionedCallStatefulPartitionedCall,dense_17022/StatefulPartitionedCall:output:0dense_17023_75415882dense_17023_75415884*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17023_layer_call_and_return_conditional_losses_75415701�
#dense_17024/StatefulPartitionedCallStatefulPartitionedCall,dense_17023/StatefulPartitionedCall:output:0dense_17024_75415887dense_17024_75415889*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17024_layer_call_and_return_conditional_losses_75415718�
#dense_17025/StatefulPartitionedCallStatefulPartitionedCall,dense_17024/StatefulPartitionedCall:output:0dense_17025_75415892dense_17025_75415894*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17025_layer_call_and_return_conditional_losses_75415734�
#dense_17026/StatefulPartitionedCallStatefulPartitionedCall,dense_17025/StatefulPartitionedCall:output:0dense_17026_75415897dense_17026_75415899*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17026_layer_call_and_return_conditional_losses_75415751�
#dense_17027/StatefulPartitionedCallStatefulPartitionedCall,dense_17026/StatefulPartitionedCall:output:0dense_17027_75415902dense_17027_75415904*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17027_layer_call_and_return_conditional_losses_75415767�
#dense_17028/StatefulPartitionedCallStatefulPartitionedCall,dense_17027/StatefulPartitionedCall:output:0dense_17028_75415907dense_17028_75415909*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17028_layer_call_and_return_conditional_losses_75415784�
#dense_17029/StatefulPartitionedCallStatefulPartitionedCall,dense_17028/StatefulPartitionedCall:output:0dense_17029_75415912dense_17029_75415914*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17029_layer_call_and_return_conditional_losses_75415800{
IdentityIdentity,dense_17029/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(�
NoOpNoOp$^dense_17020/StatefulPartitionedCall$^dense_17021/StatefulPartitionedCall$^dense_17022/StatefulPartitionedCall$^dense_17023/StatefulPartitionedCall$^dense_17024/StatefulPartitionedCall$^dense_17025/StatefulPartitionedCall$^dense_17026/StatefulPartitionedCall$^dense_17027/StatefulPartitionedCall$^dense_17028/StatefulPartitionedCall$^dense_17029/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������(: : : : : : : : : : : : : : : : : : : : 2J
#dense_17020/StatefulPartitionedCall#dense_17020/StatefulPartitionedCall2J
#dense_17021/StatefulPartitionedCall#dense_17021/StatefulPartitionedCall2J
#dense_17022/StatefulPartitionedCall#dense_17022/StatefulPartitionedCall2J
#dense_17023/StatefulPartitionedCall#dense_17023/StatefulPartitionedCall2J
#dense_17024/StatefulPartitionedCall#dense_17024/StatefulPartitionedCall2J
#dense_17025/StatefulPartitionedCall#dense_17025/StatefulPartitionedCall2J
#dense_17026/StatefulPartitionedCall#dense_17026/StatefulPartitionedCall2J
#dense_17027/StatefulPartitionedCall#dense_17027/StatefulPartitionedCall2J
#dense_17028/StatefulPartitionedCall#dense_17028/StatefulPartitionedCall2J
#dense_17029/StatefulPartitionedCall#dense_17029/StatefulPartitionedCall:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�
�
.__inference_dense_17029_layer_call_fn_75416712

inputs
unknown:(
	unknown_0:(
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17029_layer_call_and_return_conditional_losses_75415800o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
.__inference_dense_17028_layer_call_fn_75416692

inputs
unknown:(
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17028_layer_call_and_return_conditional_losses_75415784o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������(: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�
�
.__inference_dense_17020_layer_call_fn_75416536

inputs
unknown:(
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17020_layer_call_and_return_conditional_losses_75415652o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������(: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
��
�
!__inference__traced_save_75416889
file_prefix;
)read_disablecopyonread_dense_17020_kernel:(7
)read_1_disablecopyonread_dense_17020_bias:=
+read_2_disablecopyonread_dense_17021_kernel:(7
)read_3_disablecopyonread_dense_17021_bias:(=
+read_4_disablecopyonread_dense_17022_kernel:(
7
)read_5_disablecopyonread_dense_17022_bias:
=
+read_6_disablecopyonread_dense_17023_kernel:
(7
)read_7_disablecopyonread_dense_17023_bias:(=
+read_8_disablecopyonread_dense_17024_kernel:(7
)read_9_disablecopyonread_dense_17024_bias:>
,read_10_disablecopyonread_dense_17025_kernel:(8
*read_11_disablecopyonread_dense_17025_bias:(>
,read_12_disablecopyonread_dense_17026_kernel:(
8
*read_13_disablecopyonread_dense_17026_bias:
>
,read_14_disablecopyonread_dense_17027_kernel:
(8
*read_15_disablecopyonread_dense_17027_bias:(>
,read_16_disablecopyonread_dense_17028_kernel:(8
*read_17_disablecopyonread_dense_17028_bias:>
,read_18_disablecopyonread_dense_17029_kernel:(8
*read_19_disablecopyonread_dense_17029_bias:(-
#read_20_disablecopyonread_iteration:	 1
'read_21_disablecopyonread_learning_rate: )
read_22_disablecopyonread_total: )
read_23_disablecopyonread_count: 
savev2_const
identity_49��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: {
Read/DisableCopyOnReadDisableCopyOnRead)read_disablecopyonread_dense_17020_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp)read_disablecopyonread_dense_17020_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:(*
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:(a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:(}
Read_1/DisableCopyOnReadDisableCopyOnRead)read_1_disablecopyonread_dense_17020_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp)read_1_disablecopyonread_dense_17020_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_2/DisableCopyOnReadDisableCopyOnRead+read_2_disablecopyonread_dense_17021_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp+read_2_disablecopyonread_dense_17021_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:(*
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:(c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:(}
Read_3/DisableCopyOnReadDisableCopyOnRead)read_3_disablecopyonread_dense_17021_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp)read_3_disablecopyonread_dense_17021_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:(*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:(_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:(
Read_4/DisableCopyOnReadDisableCopyOnRead+read_4_disablecopyonread_dense_17022_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp+read_4_disablecopyonread_dense_17022_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:(
*
dtype0m

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:(
c

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes

:(
}
Read_5/DisableCopyOnReadDisableCopyOnRead)read_5_disablecopyonread_dense_17022_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp)read_5_disablecopyonread_dense_17022_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:

Read_6/DisableCopyOnReadDisableCopyOnRead+read_6_disablecopyonread_dense_17023_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp+read_6_disablecopyonread_dense_17023_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:
(*
dtype0n
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:
(e
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes

:
(}
Read_7/DisableCopyOnReadDisableCopyOnRead)read_7_disablecopyonread_dense_17023_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp)read_7_disablecopyonread_dense_17023_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:(*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:(a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:(
Read_8/DisableCopyOnReadDisableCopyOnRead+read_8_disablecopyonread_dense_17024_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp+read_8_disablecopyonread_dense_17024_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:(*
dtype0n
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:(e
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes

:(}
Read_9/DisableCopyOnReadDisableCopyOnRead)read_9_disablecopyonread_dense_17024_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp)read_9_disablecopyonread_dense_17024_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_10/DisableCopyOnReadDisableCopyOnRead,read_10_disablecopyonread_dense_17025_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp,read_10_disablecopyonread_dense_17025_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:(*
dtype0o
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:(e
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes

:(
Read_11/DisableCopyOnReadDisableCopyOnRead*read_11_disablecopyonread_dense_17025_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp*read_11_disablecopyonread_dense_17025_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:(*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:(a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:(�
Read_12/DisableCopyOnReadDisableCopyOnRead,read_12_disablecopyonread_dense_17026_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp,read_12_disablecopyonread_dense_17026_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:(
*
dtype0o
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:(
e
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes

:(

Read_13/DisableCopyOnReadDisableCopyOnRead*read_13_disablecopyonread_dense_17026_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp*read_13_disablecopyonread_dense_17026_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_14/DisableCopyOnReadDisableCopyOnRead,read_14_disablecopyonread_dense_17027_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp,read_14_disablecopyonread_dense_17027_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:
(*
dtype0o
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:
(e
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes

:
(
Read_15/DisableCopyOnReadDisableCopyOnRead*read_15_disablecopyonread_dense_17027_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp*read_15_disablecopyonread_dense_17027_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:(*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:(a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:(�
Read_16/DisableCopyOnReadDisableCopyOnRead,read_16_disablecopyonread_dense_17028_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp,read_16_disablecopyonread_dense_17028_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:(*
dtype0o
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:(e
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes

:(
Read_17/DisableCopyOnReadDisableCopyOnRead*read_17_disablecopyonread_dense_17028_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp*read_17_disablecopyonread_dense_17028_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_18/DisableCopyOnReadDisableCopyOnRead,read_18_disablecopyonread_dense_17029_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp,read_18_disablecopyonread_dense_17029_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:(*
dtype0o
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:(e
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes

:(
Read_19/DisableCopyOnReadDisableCopyOnRead*read_19_disablecopyonread_dense_17029_bias"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp*read_19_disablecopyonread_dense_17029_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:(*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:(a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:(x
Read_20/DisableCopyOnReadDisableCopyOnRead#read_20_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp#read_20_disablecopyonread_iteration^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_21/DisableCopyOnReadDisableCopyOnRead'read_21_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp'read_21_disablecopyonread_learning_rate^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_22/DisableCopyOnReadDisableCopyOnReadread_22_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOpread_22_disablecopyonread_total^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_23/DisableCopyOnReadDisableCopyOnReadread_23_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOpread_23_disablecopyonread_count^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�

value�
B�
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *'
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_48Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_49IdentityIdentity_48:output:0^NoOp*
T0*
_output_shapes
: �

NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_49Identity_49:output:0*G
_input_shapes6
4: : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
-__inference_model_3405_layer_call_fn_75416344

inputs
unknown:(
	unknown_0:
	unknown_1:(
	unknown_2:(
	unknown_3:(

	unknown_4:

	unknown_5:
(
	unknown_6:(
	unknown_7:(
	unknown_8:
	unknown_9:(

unknown_10:(

unknown_11:(


unknown_12:


unknown_13:
(

unknown_14:(

unknown_15:(

unknown_16:

unknown_17:(

unknown_18:(
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_model_3405_layer_call_and_return_conditional_losses_75415918o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������(: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�

�
I__inference_dense_17024_layer_call_and_return_conditional_losses_75415718

inputs0
matmul_readvariableop_resource:(-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�V
�
H__inference_model_3405_layer_call_and_return_conditional_losses_75416527

inputs<
*dense_17020_matmul_readvariableop_resource:(9
+dense_17020_biasadd_readvariableop_resource:<
*dense_17021_matmul_readvariableop_resource:(9
+dense_17021_biasadd_readvariableop_resource:(<
*dense_17022_matmul_readvariableop_resource:(
9
+dense_17022_biasadd_readvariableop_resource:
<
*dense_17023_matmul_readvariableop_resource:
(9
+dense_17023_biasadd_readvariableop_resource:(<
*dense_17024_matmul_readvariableop_resource:(9
+dense_17024_biasadd_readvariableop_resource:<
*dense_17025_matmul_readvariableop_resource:(9
+dense_17025_biasadd_readvariableop_resource:(<
*dense_17026_matmul_readvariableop_resource:(
9
+dense_17026_biasadd_readvariableop_resource:
<
*dense_17027_matmul_readvariableop_resource:
(9
+dense_17027_biasadd_readvariableop_resource:(<
*dense_17028_matmul_readvariableop_resource:(9
+dense_17028_biasadd_readvariableop_resource:<
*dense_17029_matmul_readvariableop_resource:(9
+dense_17029_biasadd_readvariableop_resource:(
identity��"dense_17020/BiasAdd/ReadVariableOp�!dense_17020/MatMul/ReadVariableOp�"dense_17021/BiasAdd/ReadVariableOp�!dense_17021/MatMul/ReadVariableOp�"dense_17022/BiasAdd/ReadVariableOp�!dense_17022/MatMul/ReadVariableOp�"dense_17023/BiasAdd/ReadVariableOp�!dense_17023/MatMul/ReadVariableOp�"dense_17024/BiasAdd/ReadVariableOp�!dense_17024/MatMul/ReadVariableOp�"dense_17025/BiasAdd/ReadVariableOp�!dense_17025/MatMul/ReadVariableOp�"dense_17026/BiasAdd/ReadVariableOp�!dense_17026/MatMul/ReadVariableOp�"dense_17027/BiasAdd/ReadVariableOp�!dense_17027/MatMul/ReadVariableOp�"dense_17028/BiasAdd/ReadVariableOp�!dense_17028/MatMul/ReadVariableOp�"dense_17029/BiasAdd/ReadVariableOp�!dense_17029/MatMul/ReadVariableOp�
!dense_17020/MatMul/ReadVariableOpReadVariableOp*dense_17020_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
dense_17020/MatMulMatMulinputs)dense_17020/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"dense_17020/BiasAdd/ReadVariableOpReadVariableOp+dense_17020_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_17020/BiasAddBiasAdddense_17020/MatMul:product:0*dense_17020/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
dense_17020/ReluReludense_17020/BiasAdd:output:0*
T0*'
_output_shapes
:����������
!dense_17021/MatMul/ReadVariableOpReadVariableOp*dense_17021_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
dense_17021/MatMulMatMuldense_17020/Relu:activations:0)dense_17021/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
"dense_17021/BiasAdd/ReadVariableOpReadVariableOp+dense_17021_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
dense_17021/BiasAddBiasAdddense_17021/MatMul:product:0*dense_17021/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
!dense_17022/MatMul/ReadVariableOpReadVariableOp*dense_17022_matmul_readvariableop_resource*
_output_shapes

:(
*
dtype0�
dense_17022/MatMulMatMuldense_17021/BiasAdd:output:0)dense_17022/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
"dense_17022/BiasAdd/ReadVariableOpReadVariableOp+dense_17022_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
dense_17022/BiasAddBiasAdddense_17022/MatMul:product:0*dense_17022/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
h
dense_17022/ReluReludense_17022/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
!dense_17023/MatMul/ReadVariableOpReadVariableOp*dense_17023_matmul_readvariableop_resource*
_output_shapes

:
(*
dtype0�
dense_17023/MatMulMatMuldense_17022/Relu:activations:0)dense_17023/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
"dense_17023/BiasAdd/ReadVariableOpReadVariableOp+dense_17023_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
dense_17023/BiasAddBiasAdddense_17023/MatMul:product:0*dense_17023/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
!dense_17024/MatMul/ReadVariableOpReadVariableOp*dense_17024_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
dense_17024/MatMulMatMuldense_17023/BiasAdd:output:0)dense_17024/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"dense_17024/BiasAdd/ReadVariableOpReadVariableOp+dense_17024_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_17024/BiasAddBiasAdddense_17024/MatMul:product:0*dense_17024/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
dense_17024/ReluReludense_17024/BiasAdd:output:0*
T0*'
_output_shapes
:����������
!dense_17025/MatMul/ReadVariableOpReadVariableOp*dense_17025_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
dense_17025/MatMulMatMuldense_17024/Relu:activations:0)dense_17025/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
"dense_17025/BiasAdd/ReadVariableOpReadVariableOp+dense_17025_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
dense_17025/BiasAddBiasAdddense_17025/MatMul:product:0*dense_17025/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
!dense_17026/MatMul/ReadVariableOpReadVariableOp*dense_17026_matmul_readvariableop_resource*
_output_shapes

:(
*
dtype0�
dense_17026/MatMulMatMuldense_17025/BiasAdd:output:0)dense_17026/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
"dense_17026/BiasAdd/ReadVariableOpReadVariableOp+dense_17026_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
dense_17026/BiasAddBiasAdddense_17026/MatMul:product:0*dense_17026/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
h
dense_17026/ReluReludense_17026/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
!dense_17027/MatMul/ReadVariableOpReadVariableOp*dense_17027_matmul_readvariableop_resource*
_output_shapes

:
(*
dtype0�
dense_17027/MatMulMatMuldense_17026/Relu:activations:0)dense_17027/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
"dense_17027/BiasAdd/ReadVariableOpReadVariableOp+dense_17027_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
dense_17027/BiasAddBiasAdddense_17027/MatMul:product:0*dense_17027/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
!dense_17028/MatMul/ReadVariableOpReadVariableOp*dense_17028_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
dense_17028/MatMulMatMuldense_17027/BiasAdd:output:0)dense_17028/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"dense_17028/BiasAdd/ReadVariableOpReadVariableOp+dense_17028_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_17028/BiasAddBiasAdddense_17028/MatMul:product:0*dense_17028/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
dense_17028/ReluReludense_17028/BiasAdd:output:0*
T0*'
_output_shapes
:����������
!dense_17029/MatMul/ReadVariableOpReadVariableOp*dense_17029_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
dense_17029/MatMulMatMuldense_17028/Relu:activations:0)dense_17029/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
"dense_17029/BiasAdd/ReadVariableOpReadVariableOp+dense_17029_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
dense_17029/BiasAddBiasAdddense_17029/MatMul:product:0*dense_17029/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(k
IdentityIdentitydense_17029/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������(�
NoOpNoOp#^dense_17020/BiasAdd/ReadVariableOp"^dense_17020/MatMul/ReadVariableOp#^dense_17021/BiasAdd/ReadVariableOp"^dense_17021/MatMul/ReadVariableOp#^dense_17022/BiasAdd/ReadVariableOp"^dense_17022/MatMul/ReadVariableOp#^dense_17023/BiasAdd/ReadVariableOp"^dense_17023/MatMul/ReadVariableOp#^dense_17024/BiasAdd/ReadVariableOp"^dense_17024/MatMul/ReadVariableOp#^dense_17025/BiasAdd/ReadVariableOp"^dense_17025/MatMul/ReadVariableOp#^dense_17026/BiasAdd/ReadVariableOp"^dense_17026/MatMul/ReadVariableOp#^dense_17027/BiasAdd/ReadVariableOp"^dense_17027/MatMul/ReadVariableOp#^dense_17028/BiasAdd/ReadVariableOp"^dense_17028/MatMul/ReadVariableOp#^dense_17029/BiasAdd/ReadVariableOp"^dense_17029/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������(: : : : : : : : : : : : : : : : : : : : 2H
"dense_17020/BiasAdd/ReadVariableOp"dense_17020/BiasAdd/ReadVariableOp2F
!dense_17020/MatMul/ReadVariableOp!dense_17020/MatMul/ReadVariableOp2H
"dense_17021/BiasAdd/ReadVariableOp"dense_17021/BiasAdd/ReadVariableOp2F
!dense_17021/MatMul/ReadVariableOp!dense_17021/MatMul/ReadVariableOp2H
"dense_17022/BiasAdd/ReadVariableOp"dense_17022/BiasAdd/ReadVariableOp2F
!dense_17022/MatMul/ReadVariableOp!dense_17022/MatMul/ReadVariableOp2H
"dense_17023/BiasAdd/ReadVariableOp"dense_17023/BiasAdd/ReadVariableOp2F
!dense_17023/MatMul/ReadVariableOp!dense_17023/MatMul/ReadVariableOp2H
"dense_17024/BiasAdd/ReadVariableOp"dense_17024/BiasAdd/ReadVariableOp2F
!dense_17024/MatMul/ReadVariableOp!dense_17024/MatMul/ReadVariableOp2H
"dense_17025/BiasAdd/ReadVariableOp"dense_17025/BiasAdd/ReadVariableOp2F
!dense_17025/MatMul/ReadVariableOp!dense_17025/MatMul/ReadVariableOp2H
"dense_17026/BiasAdd/ReadVariableOp"dense_17026/BiasAdd/ReadVariableOp2F
!dense_17026/MatMul/ReadVariableOp!dense_17026/MatMul/ReadVariableOp2H
"dense_17027/BiasAdd/ReadVariableOp"dense_17027/BiasAdd/ReadVariableOp2F
!dense_17027/MatMul/ReadVariableOp!dense_17027/MatMul/ReadVariableOp2H
"dense_17028/BiasAdd/ReadVariableOp"dense_17028/BiasAdd/ReadVariableOp2F
!dense_17028/MatMul/ReadVariableOp!dense_17028/MatMul/ReadVariableOp2H
"dense_17029/BiasAdd/ReadVariableOp"dense_17029/BiasAdd/ReadVariableOp2F
!dense_17029/MatMul/ReadVariableOp!dense_17029/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�

�
I__inference_dense_17020_layer_call_and_return_conditional_losses_75416547

inputs0
matmul_readvariableop_resource:(-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�

�
I__inference_dense_17028_layer_call_and_return_conditional_losses_75415784

inputs0
matmul_readvariableop_resource:(-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�
�
-__inference_model_3405_layer_call_fn_75416060

input_3406
unknown:(
	unknown_0:
	unknown_1:(
	unknown_2:(
	unknown_3:(

	unknown_4:

	unknown_5:
(
	unknown_6:(
	unknown_7:(
	unknown_8:
	unknown_9:(

unknown_10:(

unknown_11:(


unknown_12:


unknown_13:
(

unknown_14:(

unknown_15:(

unknown_16:

unknown_17:(

unknown_18:(
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
input_3406unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_model_3405_layer_call_and_return_conditional_losses_75416017o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������(: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:���������(
$
_user_specified_name
input_3406
�

�
I__inference_dense_17026_layer_call_and_return_conditional_losses_75415751

inputs0
matmul_readvariableop_resource:(
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�
�
&__inference_signature_wrapper_75416299

input_3406
unknown:(
	unknown_0:
	unknown_1:(
	unknown_2:(
	unknown_3:(

	unknown_4:

	unknown_5:
(
	unknown_6:(
	unknown_7:(
	unknown_8:
	unknown_9:(

unknown_10:(

unknown_11:(


unknown_12:


unknown_13:
(

unknown_14:(

unknown_15:(

unknown_16:

unknown_17:(

unknown_18:(
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
input_3406unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__wrapped_model_75415637o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������(: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:���������(
$
_user_specified_name
input_3406
�	
�
I__inference_dense_17027_layer_call_and_return_conditional_losses_75415767

inputs0
matmul_readvariableop_resource:
(-
biasadd_readvariableop_resource:(
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
.__inference_dense_17027_layer_call_fn_75416673

inputs
unknown:
(
	unknown_0:(
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17027_layer_call_and_return_conditional_losses_75415767o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�g
�
$__inference__traced_restore_75416971
file_prefix5
#assignvariableop_dense_17020_kernel:(1
#assignvariableop_1_dense_17020_bias:7
%assignvariableop_2_dense_17021_kernel:(1
#assignvariableop_3_dense_17021_bias:(7
%assignvariableop_4_dense_17022_kernel:(
1
#assignvariableop_5_dense_17022_bias:
7
%assignvariableop_6_dense_17023_kernel:
(1
#assignvariableop_7_dense_17023_bias:(7
%assignvariableop_8_dense_17024_kernel:(1
#assignvariableop_9_dense_17024_bias:8
&assignvariableop_10_dense_17025_kernel:(2
$assignvariableop_11_dense_17025_bias:(8
&assignvariableop_12_dense_17026_kernel:(
2
$assignvariableop_13_dense_17026_bias:
8
&assignvariableop_14_dense_17027_kernel:
(2
$assignvariableop_15_dense_17027_bias:(8
&assignvariableop_16_dense_17028_kernel:(2
$assignvariableop_17_dense_17028_bias:8
&assignvariableop_18_dense_17029_kernel:(2
$assignvariableop_19_dense_17029_bias:('
assignvariableop_20_iteration:	 +
!assignvariableop_21_learning_rate: #
assignvariableop_22_total: #
assignvariableop_23_count: 
identity_25��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�

value�
B�
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp#assignvariableop_dense_17020_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp#assignvariableop_1_dense_17020_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp%assignvariableop_2_dense_17021_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp#assignvariableop_3_dense_17021_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp%assignvariableop_4_dense_17022_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp#assignvariableop_5_dense_17022_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp%assignvariableop_6_dense_17023_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp#assignvariableop_7_dense_17023_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp%assignvariableop_8_dense_17024_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp#assignvariableop_9_dense_17024_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp&assignvariableop_10_dense_17025_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp$assignvariableop_11_dense_17025_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp&assignvariableop_12_dense_17026_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp$assignvariableop_13_dense_17026_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp&assignvariableop_14_dense_17027_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp$assignvariableop_15_dense_17027_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp&assignvariableop_16_dense_17028_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp$assignvariableop_17_dense_17028_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp&assignvariableop_18_dense_17029_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp$assignvariableop_19_dense_17029_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_iterationIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp!assignvariableop_21_learning_rateIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_totalIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_countIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_25IdentityIdentity_24:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_25Identity_25:output:0*E
_input_shapes4
2: : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�7
�	
H__inference_model_3405_layer_call_and_return_conditional_losses_75415861

input_3406&
dense_17020_75415810:("
dense_17020_75415812:&
dense_17021_75415815:("
dense_17021_75415817:(&
dense_17022_75415820:(
"
dense_17022_75415822:
&
dense_17023_75415825:
("
dense_17023_75415827:(&
dense_17024_75415830:("
dense_17024_75415832:&
dense_17025_75415835:("
dense_17025_75415837:(&
dense_17026_75415840:(
"
dense_17026_75415842:
&
dense_17027_75415845:
("
dense_17027_75415847:(&
dense_17028_75415850:("
dense_17028_75415852:&
dense_17029_75415855:("
dense_17029_75415857:(
identity��#dense_17020/StatefulPartitionedCall�#dense_17021/StatefulPartitionedCall�#dense_17022/StatefulPartitionedCall�#dense_17023/StatefulPartitionedCall�#dense_17024/StatefulPartitionedCall�#dense_17025/StatefulPartitionedCall�#dense_17026/StatefulPartitionedCall�#dense_17027/StatefulPartitionedCall�#dense_17028/StatefulPartitionedCall�#dense_17029/StatefulPartitionedCall�
#dense_17020/StatefulPartitionedCallStatefulPartitionedCall
input_3406dense_17020_75415810dense_17020_75415812*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17020_layer_call_and_return_conditional_losses_75415652�
#dense_17021/StatefulPartitionedCallStatefulPartitionedCall,dense_17020/StatefulPartitionedCall:output:0dense_17021_75415815dense_17021_75415817*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17021_layer_call_and_return_conditional_losses_75415668�
#dense_17022/StatefulPartitionedCallStatefulPartitionedCall,dense_17021/StatefulPartitionedCall:output:0dense_17022_75415820dense_17022_75415822*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17022_layer_call_and_return_conditional_losses_75415685�
#dense_17023/StatefulPartitionedCallStatefulPartitionedCall,dense_17022/StatefulPartitionedCall:output:0dense_17023_75415825dense_17023_75415827*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17023_layer_call_and_return_conditional_losses_75415701�
#dense_17024/StatefulPartitionedCallStatefulPartitionedCall,dense_17023/StatefulPartitionedCall:output:0dense_17024_75415830dense_17024_75415832*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17024_layer_call_and_return_conditional_losses_75415718�
#dense_17025/StatefulPartitionedCallStatefulPartitionedCall,dense_17024/StatefulPartitionedCall:output:0dense_17025_75415835dense_17025_75415837*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17025_layer_call_and_return_conditional_losses_75415734�
#dense_17026/StatefulPartitionedCallStatefulPartitionedCall,dense_17025/StatefulPartitionedCall:output:0dense_17026_75415840dense_17026_75415842*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17026_layer_call_and_return_conditional_losses_75415751�
#dense_17027/StatefulPartitionedCallStatefulPartitionedCall,dense_17026/StatefulPartitionedCall:output:0dense_17027_75415845dense_17027_75415847*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17027_layer_call_and_return_conditional_losses_75415767�
#dense_17028/StatefulPartitionedCallStatefulPartitionedCall,dense_17027/StatefulPartitionedCall:output:0dense_17028_75415850dense_17028_75415852*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17028_layer_call_and_return_conditional_losses_75415784�
#dense_17029/StatefulPartitionedCallStatefulPartitionedCall,dense_17028/StatefulPartitionedCall:output:0dense_17029_75415855dense_17029_75415857*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_17029_layer_call_and_return_conditional_losses_75415800{
IdentityIdentity,dense_17029/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(�
NoOpNoOp$^dense_17020/StatefulPartitionedCall$^dense_17021/StatefulPartitionedCall$^dense_17022/StatefulPartitionedCall$^dense_17023/StatefulPartitionedCall$^dense_17024/StatefulPartitionedCall$^dense_17025/StatefulPartitionedCall$^dense_17026/StatefulPartitionedCall$^dense_17027/StatefulPartitionedCall$^dense_17028/StatefulPartitionedCall$^dense_17029/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������(: : : : : : : : : : : : : : : : : : : : 2J
#dense_17020/StatefulPartitionedCall#dense_17020/StatefulPartitionedCall2J
#dense_17021/StatefulPartitionedCall#dense_17021/StatefulPartitionedCall2J
#dense_17022/StatefulPartitionedCall#dense_17022/StatefulPartitionedCall2J
#dense_17023/StatefulPartitionedCall#dense_17023/StatefulPartitionedCall2J
#dense_17024/StatefulPartitionedCall#dense_17024/StatefulPartitionedCall2J
#dense_17025/StatefulPartitionedCall#dense_17025/StatefulPartitionedCall2J
#dense_17026/StatefulPartitionedCall#dense_17026/StatefulPartitionedCall2J
#dense_17027/StatefulPartitionedCall#dense_17027/StatefulPartitionedCall2J
#dense_17028/StatefulPartitionedCall#dense_17028/StatefulPartitionedCall2J
#dense_17029/StatefulPartitionedCall#dense_17029/StatefulPartitionedCall:S O
'
_output_shapes
:���������(
$
_user_specified_name
input_3406"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
A

input_34063
serving_default_input_3406:0���������(?
dense_170290
StatefulPartitionedCall:0���������(tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
layer_with_weights-9
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias"
_tf_keras_layer
�
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

+kernel
,bias"
_tf_keras_layer
�
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

3kernel
4bias"
_tf_keras_layer
�
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

;kernel
<bias"
_tf_keras_layer
�
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses

Ckernel
Dbias"
_tf_keras_layer
�
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses

Kkernel
Lbias"
_tf_keras_layer
�
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses

Skernel
Tbias"
_tf_keras_layer
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses

[kernel
\bias"
_tf_keras_layer
�
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses

ckernel
dbias"
_tf_keras_layer
�
0
1
#2
$3
+4
,5
36
47
;8
<9
C10
D11
K12
L13
S14
T15
[16
\17
c18
d19"
trackable_list_wrapper
�
0
1
#2
$3
+4
,5
36
47
;8
<9
C10
D11
K12
L13
S14
T15
[16
\17
c18
d19"
trackable_list_wrapper
 "
trackable_list_wrapper
�
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
jtrace_0
ktrace_1
ltrace_2
mtrace_32�
-__inference_model_3405_layer_call_fn_75415961
-__inference_model_3405_layer_call_fn_75416060
-__inference_model_3405_layer_call_fn_75416344
-__inference_model_3405_layer_call_fn_75416389�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zjtrace_0zktrace_1zltrace_2zmtrace_3
�
ntrace_0
otrace_1
ptrace_2
qtrace_32�
H__inference_model_3405_layer_call_and_return_conditional_losses_75415807
H__inference_model_3405_layer_call_and_return_conditional_losses_75415861
H__inference_model_3405_layer_call_and_return_conditional_losses_75416458
H__inference_model_3405_layer_call_and_return_conditional_losses_75416527�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zntrace_0zotrace_1zptrace_2zqtrace_3
�B�
#__inference__wrapped_model_75415637
input_3406"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
j
r
_variables
s_iterations
t_learning_rate
u_update_step_xla"
experimentalOptimizer
,
vserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
|trace_02�
.__inference_dense_17020_layer_call_fn_75416536�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z|trace_0
�
}trace_02�
I__inference_dense_17020_layer_call_and_return_conditional_losses_75416547�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z}trace_0
$:"(2dense_17020/kernel
:2dense_17020/bias
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
~non_trainable_variables

layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_dense_17021_layer_call_fn_75416556�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_dense_17021_layer_call_and_return_conditional_losses_75416566�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
$:"(2dense_17021/kernel
:(2dense_17021/bias
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_dense_17022_layer_call_fn_75416575�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_dense_17022_layer_call_and_return_conditional_losses_75416586�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
$:"(
2dense_17022/kernel
:
2dense_17022/bias
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_dense_17023_layer_call_fn_75416595�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_dense_17023_layer_call_and_return_conditional_losses_75416605�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
$:"
(2dense_17023/kernel
:(2dense_17023/bias
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_dense_17024_layer_call_fn_75416614�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_dense_17024_layer_call_and_return_conditional_losses_75416625�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
$:"(2dense_17024/kernel
:2dense_17024/bias
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_dense_17025_layer_call_fn_75416634�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_dense_17025_layer_call_and_return_conditional_losses_75416644�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
$:"(2dense_17025/kernel
:(2dense_17025/bias
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_dense_17026_layer_call_fn_75416653�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_dense_17026_layer_call_and_return_conditional_losses_75416664�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
$:"(
2dense_17026/kernel
:
2dense_17026/bias
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_dense_17027_layer_call_fn_75416673�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_dense_17027_layer_call_and_return_conditional_losses_75416683�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
$:"
(2dense_17027/kernel
:(2dense_17027/bias
.
[0
\1"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_dense_17028_layer_call_fn_75416692�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_dense_17028_layer_call_and_return_conditional_losses_75416703�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
$:"(2dense_17028/kernel
:2dense_17028/bias
.
c0
d1"
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_dense_17029_layer_call_fn_75416712�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_dense_17029_layer_call_and_return_conditional_losses_75416722�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
$:"(2dense_17029/kernel
:(2dense_17029/bias
 "
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_model_3405_layer_call_fn_75415961
input_3406"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_model_3405_layer_call_fn_75416060
input_3406"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_model_3405_layer_call_fn_75416344inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_model_3405_layer_call_fn_75416389inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_model_3405_layer_call_and_return_conditional_losses_75415807
input_3406"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_model_3405_layer_call_and_return_conditional_losses_75415861
input_3406"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_model_3405_layer_call_and_return_conditional_losses_75416458inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_model_3405_layer_call_and_return_conditional_losses_75416527inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
'
s0"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
&__inference_signature_wrapper_75416299
input_3406"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_dense_17020_layer_call_fn_75416536inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_dense_17020_layer_call_and_return_conditional_losses_75416547inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_dense_17021_layer_call_fn_75416556inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_dense_17021_layer_call_and_return_conditional_losses_75416566inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_dense_17022_layer_call_fn_75416575inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_dense_17022_layer_call_and_return_conditional_losses_75416586inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_dense_17023_layer_call_fn_75416595inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_dense_17023_layer_call_and_return_conditional_losses_75416605inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_dense_17024_layer_call_fn_75416614inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_dense_17024_layer_call_and_return_conditional_losses_75416625inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_dense_17025_layer_call_fn_75416634inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_dense_17025_layer_call_and_return_conditional_losses_75416644inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_dense_17026_layer_call_fn_75416653inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_dense_17026_layer_call_and_return_conditional_losses_75416664inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_dense_17027_layer_call_fn_75416673inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_dense_17027_layer_call_and_return_conditional_losses_75416683inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_dense_17028_layer_call_fn_75416692inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_dense_17028_layer_call_and_return_conditional_losses_75416703inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_dense_17029_layer_call_fn_75416712inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_dense_17029_layer_call_and_return_conditional_losses_75416722inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count�
#__inference__wrapped_model_75415637�#$+,34;<CDKLST[\cd3�0
)�&
$�!

input_3406���������(
� "9�6
4
dense_17029%�"
dense_17029���������(�
I__inference_dense_17020_layer_call_and_return_conditional_losses_75416547c/�,
%�"
 �
inputs���������(
� ",�)
"�
tensor_0���������
� �
.__inference_dense_17020_layer_call_fn_75416536X/�,
%�"
 �
inputs���������(
� "!�
unknown����������
I__inference_dense_17021_layer_call_and_return_conditional_losses_75416566c#$/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������(
� �
.__inference_dense_17021_layer_call_fn_75416556X#$/�,
%�"
 �
inputs���������
� "!�
unknown���������(�
I__inference_dense_17022_layer_call_and_return_conditional_losses_75416586c+,/�,
%�"
 �
inputs���������(
� ",�)
"�
tensor_0���������

� �
.__inference_dense_17022_layer_call_fn_75416575X+,/�,
%�"
 �
inputs���������(
� "!�
unknown���������
�
I__inference_dense_17023_layer_call_and_return_conditional_losses_75416605c34/�,
%�"
 �
inputs���������

� ",�)
"�
tensor_0���������(
� �
.__inference_dense_17023_layer_call_fn_75416595X34/�,
%�"
 �
inputs���������

� "!�
unknown���������(�
I__inference_dense_17024_layer_call_and_return_conditional_losses_75416625c;</�,
%�"
 �
inputs���������(
� ",�)
"�
tensor_0���������
� �
.__inference_dense_17024_layer_call_fn_75416614X;</�,
%�"
 �
inputs���������(
� "!�
unknown����������
I__inference_dense_17025_layer_call_and_return_conditional_losses_75416644cCD/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������(
� �
.__inference_dense_17025_layer_call_fn_75416634XCD/�,
%�"
 �
inputs���������
� "!�
unknown���������(�
I__inference_dense_17026_layer_call_and_return_conditional_losses_75416664cKL/�,
%�"
 �
inputs���������(
� ",�)
"�
tensor_0���������

� �
.__inference_dense_17026_layer_call_fn_75416653XKL/�,
%�"
 �
inputs���������(
� "!�
unknown���������
�
I__inference_dense_17027_layer_call_and_return_conditional_losses_75416683cST/�,
%�"
 �
inputs���������

� ",�)
"�
tensor_0���������(
� �
.__inference_dense_17027_layer_call_fn_75416673XST/�,
%�"
 �
inputs���������

� "!�
unknown���������(�
I__inference_dense_17028_layer_call_and_return_conditional_losses_75416703c[\/�,
%�"
 �
inputs���������(
� ",�)
"�
tensor_0���������
� �
.__inference_dense_17028_layer_call_fn_75416692X[\/�,
%�"
 �
inputs���������(
� "!�
unknown����������
I__inference_dense_17029_layer_call_and_return_conditional_losses_75416722ccd/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������(
� �
.__inference_dense_17029_layer_call_fn_75416712Xcd/�,
%�"
 �
inputs���������
� "!�
unknown���������(�
H__inference_model_3405_layer_call_and_return_conditional_losses_75415807�#$+,34;<CDKLST[\cd;�8
1�.
$�!

input_3406���������(
p

 
� ",�)
"�
tensor_0���������(
� �
H__inference_model_3405_layer_call_and_return_conditional_losses_75415861�#$+,34;<CDKLST[\cd;�8
1�.
$�!

input_3406���������(
p 

 
� ",�)
"�
tensor_0���������(
� �
H__inference_model_3405_layer_call_and_return_conditional_losses_75416458}#$+,34;<CDKLST[\cd7�4
-�*
 �
inputs���������(
p

 
� ",�)
"�
tensor_0���������(
� �
H__inference_model_3405_layer_call_and_return_conditional_losses_75416527}#$+,34;<CDKLST[\cd7�4
-�*
 �
inputs���������(
p 

 
� ",�)
"�
tensor_0���������(
� �
-__inference_model_3405_layer_call_fn_75415961v#$+,34;<CDKLST[\cd;�8
1�.
$�!

input_3406���������(
p

 
� "!�
unknown���������(�
-__inference_model_3405_layer_call_fn_75416060v#$+,34;<CDKLST[\cd;�8
1�.
$�!

input_3406���������(
p 

 
� "!�
unknown���������(�
-__inference_model_3405_layer_call_fn_75416344r#$+,34;<CDKLST[\cd7�4
-�*
 �
inputs���������(
p

 
� "!�
unknown���������(�
-__inference_model_3405_layer_call_fn_75416389r#$+,34;<CDKLST[\cd7�4
-�*
 �
inputs���������(
p 

 
� "!�
unknown���������(�
&__inference_signature_wrapper_75416299�#$+,34;<CDKLST[\cdA�>
� 
7�4
2

input_3406$�!

input_3406���������("9�6
4
dense_17029%�"
dense_17029���������(