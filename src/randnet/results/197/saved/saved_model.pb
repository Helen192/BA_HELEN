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
dense_17979/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*!
shared_namedense_17979/bias
q
$dense_17979/bias/Read/ReadVariableOpReadVariableOpdense_17979/bias*
_output_shapes
:(*
dtype0
�
dense_17979/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*#
shared_namedense_17979/kernel
y
&dense_17979/kernel/Read/ReadVariableOpReadVariableOpdense_17979/kernel*
_output_shapes

:(*
dtype0
x
dense_17978/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namedense_17978/bias
q
$dense_17978/bias/Read/ReadVariableOpReadVariableOpdense_17978/bias*
_output_shapes
:*
dtype0
�
dense_17978/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*#
shared_namedense_17978/kernel
y
&dense_17978/kernel/Read/ReadVariableOpReadVariableOpdense_17978/kernel*
_output_shapes

:(*
dtype0
x
dense_17977/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*!
shared_namedense_17977/bias
q
$dense_17977/bias/Read/ReadVariableOpReadVariableOpdense_17977/bias*
_output_shapes
:(*
dtype0
�
dense_17977/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
(*#
shared_namedense_17977/kernel
y
&dense_17977/kernel/Read/ReadVariableOpReadVariableOpdense_17977/kernel*
_output_shapes

:
(*
dtype0
x
dense_17976/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_namedense_17976/bias
q
$dense_17976/bias/Read/ReadVariableOpReadVariableOpdense_17976/bias*
_output_shapes
:
*
dtype0
�
dense_17976/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(
*#
shared_namedense_17976/kernel
y
&dense_17976/kernel/Read/ReadVariableOpReadVariableOpdense_17976/kernel*
_output_shapes

:(
*
dtype0
x
dense_17975/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*!
shared_namedense_17975/bias
q
$dense_17975/bias/Read/ReadVariableOpReadVariableOpdense_17975/bias*
_output_shapes
:(*
dtype0
�
dense_17975/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*#
shared_namedense_17975/kernel
y
&dense_17975/kernel/Read/ReadVariableOpReadVariableOpdense_17975/kernel*
_output_shapes

:(*
dtype0
x
dense_17974/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namedense_17974/bias
q
$dense_17974/bias/Read/ReadVariableOpReadVariableOpdense_17974/bias*
_output_shapes
:*
dtype0
�
dense_17974/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*#
shared_namedense_17974/kernel
y
&dense_17974/kernel/Read/ReadVariableOpReadVariableOpdense_17974/kernel*
_output_shapes

:(*
dtype0
x
dense_17973/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*!
shared_namedense_17973/bias
q
$dense_17973/bias/Read/ReadVariableOpReadVariableOpdense_17973/bias*
_output_shapes
:(*
dtype0
�
dense_17973/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
(*#
shared_namedense_17973/kernel
y
&dense_17973/kernel/Read/ReadVariableOpReadVariableOpdense_17973/kernel*
_output_shapes

:
(*
dtype0
x
dense_17972/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_namedense_17972/bias
q
$dense_17972/bias/Read/ReadVariableOpReadVariableOpdense_17972/bias*
_output_shapes
:
*
dtype0
�
dense_17972/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(
*#
shared_namedense_17972/kernel
y
&dense_17972/kernel/Read/ReadVariableOpReadVariableOpdense_17972/kernel*
_output_shapes

:(
*
dtype0
x
dense_17971/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*!
shared_namedense_17971/bias
q
$dense_17971/bias/Read/ReadVariableOpReadVariableOpdense_17971/bias*
_output_shapes
:(*
dtype0
�
dense_17971/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*#
shared_namedense_17971/kernel
y
&dense_17971/kernel/Read/ReadVariableOpReadVariableOpdense_17971/kernel*
_output_shapes

:(*
dtype0
x
dense_17970/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namedense_17970/bias
q
$dense_17970/bias/Read/ReadVariableOpReadVariableOpdense_17970/bias*
_output_shapes
:*
dtype0
�
dense_17970/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*#
shared_namedense_17970/kernel
y
&dense_17970/kernel/Read/ReadVariableOpReadVariableOpdense_17970/kernel*
_output_shapes

:(*
dtype0
}
serving_default_input_3596Placeholder*'
_output_shapes
:���������(*
dtype0*
shape:���������(
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3596dense_17970/kerneldense_17970/biasdense_17971/kerneldense_17971/biasdense_17972/kerneldense_17972/biasdense_17973/kerneldense_17973/biasdense_17974/kerneldense_17974/biasdense_17975/kerneldense_17975/biasdense_17976/kerneldense_17976/biasdense_17977/kerneldense_17977/biasdense_17978/kerneldense_17978/biasdense_17979/kerneldense_17979/bias* 
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
&__inference_signature_wrapper_77569094

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
VARIABLE_VALUEdense_17970/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEdense_17970/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_17971/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEdense_17971/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_17972/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEdense_17972/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_17973/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEdense_17973/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_17974/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEdense_17974/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_17975/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEdense_17975/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_17976/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEdense_17976/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_17977/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEdense_17977/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_17978/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEdense_17978/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_17979/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEdense_17979/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_17970/kerneldense_17970/biasdense_17971/kerneldense_17971/biasdense_17972/kerneldense_17972/biasdense_17973/kerneldense_17973/biasdense_17974/kerneldense_17974/biasdense_17975/kerneldense_17975/biasdense_17976/kerneldense_17976/biasdense_17977/kerneldense_17977/biasdense_17978/kerneldense_17978/biasdense_17979/kerneldense_17979/bias	iterationlearning_ratetotalcountConst*%
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
!__inference__traced_save_77569684
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_17970/kerneldense_17970/biasdense_17971/kerneldense_17971/biasdense_17972/kerneldense_17972/biasdense_17973/kerneldense_17973/biasdense_17974/kerneldense_17974/biasdense_17975/kerneldense_17975/biasdense_17976/kerneldense_17976/biasdense_17977/kerneldense_17977/biasdense_17978/kerneldense_17978/biasdense_17979/kerneldense_17979/bias	iterationlearning_ratetotalcount*$
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
$__inference__traced_restore_77569766��
�	
�
I__inference_dense_17979_layer_call_and_return_conditional_losses_77568595

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
I__inference_dense_17974_layer_call_and_return_conditional_losses_77569420

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
I__inference_dense_17979_layer_call_and_return_conditional_losses_77569517

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
.__inference_dense_17979_layer_call_fn_77569507

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
I__inference_dense_17979_layer_call_and_return_conditional_losses_77568595o
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
�

�
I__inference_dense_17970_layer_call_and_return_conditional_losses_77569342

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
I__inference_dense_17972_layer_call_and_return_conditional_losses_77568480

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
��
�
!__inference__traced_save_77569684
file_prefix;
)read_disablecopyonread_dense_17970_kernel:(7
)read_1_disablecopyonread_dense_17970_bias:=
+read_2_disablecopyonread_dense_17971_kernel:(7
)read_3_disablecopyonread_dense_17971_bias:(=
+read_4_disablecopyonread_dense_17972_kernel:(
7
)read_5_disablecopyonread_dense_17972_bias:
=
+read_6_disablecopyonread_dense_17973_kernel:
(7
)read_7_disablecopyonread_dense_17973_bias:(=
+read_8_disablecopyonread_dense_17974_kernel:(7
)read_9_disablecopyonread_dense_17974_bias:>
,read_10_disablecopyonread_dense_17975_kernel:(8
*read_11_disablecopyonread_dense_17975_bias:(>
,read_12_disablecopyonread_dense_17976_kernel:(
8
*read_13_disablecopyonread_dense_17976_bias:
>
,read_14_disablecopyonread_dense_17977_kernel:
(8
*read_15_disablecopyonread_dense_17977_bias:(>
,read_16_disablecopyonread_dense_17978_kernel:(8
*read_17_disablecopyonread_dense_17978_bias:>
,read_18_disablecopyonread_dense_17979_kernel:(8
*read_19_disablecopyonread_dense_17979_bias:(-
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
Read/DisableCopyOnReadDisableCopyOnRead)read_disablecopyonread_dense_17970_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp)read_disablecopyonread_dense_17970_kernel^Read/DisableCopyOnRead"/device:CPU:0*
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
Read_1/DisableCopyOnReadDisableCopyOnRead)read_1_disablecopyonread_dense_17970_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp)read_1_disablecopyonread_dense_17970_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
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
Read_2/DisableCopyOnReadDisableCopyOnRead+read_2_disablecopyonread_dense_17971_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp+read_2_disablecopyonread_dense_17971_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
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
Read_3/DisableCopyOnReadDisableCopyOnRead)read_3_disablecopyonread_dense_17971_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp)read_3_disablecopyonread_dense_17971_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
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
Read_4/DisableCopyOnReadDisableCopyOnRead+read_4_disablecopyonread_dense_17972_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp+read_4_disablecopyonread_dense_17972_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
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
Read_5/DisableCopyOnReadDisableCopyOnRead)read_5_disablecopyonread_dense_17972_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp)read_5_disablecopyonread_dense_17972_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
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
Read_6/DisableCopyOnReadDisableCopyOnRead+read_6_disablecopyonread_dense_17973_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp+read_6_disablecopyonread_dense_17973_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
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
Read_7/DisableCopyOnReadDisableCopyOnRead)read_7_disablecopyonread_dense_17973_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp)read_7_disablecopyonread_dense_17973_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
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
Read_8/DisableCopyOnReadDisableCopyOnRead+read_8_disablecopyonread_dense_17974_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp+read_8_disablecopyonread_dense_17974_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
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
Read_9/DisableCopyOnReadDisableCopyOnRead)read_9_disablecopyonread_dense_17974_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp)read_9_disablecopyonread_dense_17974_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
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
Read_10/DisableCopyOnReadDisableCopyOnRead,read_10_disablecopyonread_dense_17975_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp,read_10_disablecopyonread_dense_17975_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*
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
Read_11/DisableCopyOnReadDisableCopyOnRead*read_11_disablecopyonread_dense_17975_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp*read_11_disablecopyonread_dense_17975_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
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
Read_12/DisableCopyOnReadDisableCopyOnRead,read_12_disablecopyonread_dense_17976_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp,read_12_disablecopyonread_dense_17976_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
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
Read_13/DisableCopyOnReadDisableCopyOnRead*read_13_disablecopyonread_dense_17976_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp*read_13_disablecopyonread_dense_17976_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
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
Read_14/DisableCopyOnReadDisableCopyOnRead,read_14_disablecopyonread_dense_17977_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp,read_14_disablecopyonread_dense_17977_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*
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
Read_15/DisableCopyOnReadDisableCopyOnRead*read_15_disablecopyonread_dense_17977_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp*read_15_disablecopyonread_dense_17977_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
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
Read_16/DisableCopyOnReadDisableCopyOnRead,read_16_disablecopyonread_dense_17978_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp,read_16_disablecopyonread_dense_17978_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*
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
Read_17/DisableCopyOnReadDisableCopyOnRead*read_17_disablecopyonread_dense_17978_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp*read_17_disablecopyonread_dense_17978_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
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
Read_18/DisableCopyOnReadDisableCopyOnRead,read_18_disablecopyonread_dense_17979_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp,read_18_disablecopyonread_dense_17979_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*
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
Read_19/DisableCopyOnReadDisableCopyOnRead*read_19_disablecopyonread_dense_17979_bias"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp*read_19_disablecopyonread_dense_17979_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
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
�
I__inference_dense_17975_layer_call_and_return_conditional_losses_77569439

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
�
�
.__inference_dense_17971_layer_call_fn_77569351

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
I__inference_dense_17971_layer_call_and_return_conditional_losses_77568463o
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
�

�
I__inference_dense_17976_layer_call_and_return_conditional_losses_77569459

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
I__inference_dense_17972_layer_call_and_return_conditional_losses_77569381

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
.__inference_dense_17975_layer_call_fn_77569429

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
I__inference_dense_17975_layer_call_and_return_conditional_losses_77568529o
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
�	
�
I__inference_dense_17971_layer_call_and_return_conditional_losses_77568463

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
.__inference_dense_17977_layer_call_fn_77569468

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
I__inference_dense_17977_layer_call_and_return_conditional_losses_77568562o
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
$__inference__traced_restore_77569766
file_prefix5
#assignvariableop_dense_17970_kernel:(1
#assignvariableop_1_dense_17970_bias:7
%assignvariableop_2_dense_17971_kernel:(1
#assignvariableop_3_dense_17971_bias:(7
%assignvariableop_4_dense_17972_kernel:(
1
#assignvariableop_5_dense_17972_bias:
7
%assignvariableop_6_dense_17973_kernel:
(1
#assignvariableop_7_dense_17973_bias:(7
%assignvariableop_8_dense_17974_kernel:(1
#assignvariableop_9_dense_17974_bias:8
&assignvariableop_10_dense_17975_kernel:(2
$assignvariableop_11_dense_17975_bias:(8
&assignvariableop_12_dense_17976_kernel:(
2
$assignvariableop_13_dense_17976_bias:
8
&assignvariableop_14_dense_17977_kernel:
(2
$assignvariableop_15_dense_17977_bias:(8
&assignvariableop_16_dense_17978_kernel:(2
$assignvariableop_17_dense_17978_bias:8
&assignvariableop_18_dense_17979_kernel:(2
$assignvariableop_19_dense_17979_bias:('
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
AssignVariableOpAssignVariableOp#assignvariableop_dense_17970_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp#assignvariableop_1_dense_17970_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp%assignvariableop_2_dense_17971_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp#assignvariableop_3_dense_17971_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp%assignvariableop_4_dense_17972_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp#assignvariableop_5_dense_17972_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp%assignvariableop_6_dense_17973_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp#assignvariableop_7_dense_17973_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp%assignvariableop_8_dense_17974_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp#assignvariableop_9_dense_17974_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp&assignvariableop_10_dense_17975_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp$assignvariableop_11_dense_17975_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp&assignvariableop_12_dense_17976_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp$assignvariableop_13_dense_17976_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp&assignvariableop_14_dense_17977_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp$assignvariableop_15_dense_17977_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp&assignvariableop_16_dense_17978_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp$assignvariableop_17_dense_17978_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp&assignvariableop_18_dense_17979_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp$assignvariableop_19_dense_17979_biasIdentity_19:output:0"/device:CPU:0*&
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
�

�
I__inference_dense_17974_layer_call_and_return_conditional_losses_77568513

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
�
&__inference_signature_wrapper_77569094

input_3596
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
input_3596unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
#__inference__wrapped_model_77568432o
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
input_3596
�	
�
I__inference_dense_17977_layer_call_and_return_conditional_losses_77569478

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
.__inference_dense_17976_layer_call_fn_77569448

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
I__inference_dense_17976_layer_call_and_return_conditional_losses_77568546o
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
�
-__inference_model_3595_layer_call_fn_77568855

input_3596
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
input_3596unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
H__inference_model_3595_layer_call_and_return_conditional_losses_77568812o
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
input_3596
�
�
.__inference_dense_17974_layer_call_fn_77569409

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
I__inference_dense_17974_layer_call_and_return_conditional_losses_77568513o
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
�
-__inference_model_3595_layer_call_fn_77569139

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
H__inference_model_3595_layer_call_and_return_conditional_losses_77568713o
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
I__inference_dense_17975_layer_call_and_return_conditional_losses_77568529

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
�
�
.__inference_dense_17970_layer_call_fn_77569331

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
I__inference_dense_17970_layer_call_and_return_conditional_losses_77568447o
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
�

�
I__inference_dense_17976_layer_call_and_return_conditional_losses_77568546

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
I__inference_dense_17973_layer_call_and_return_conditional_losses_77569400

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
�h
�
#__inference__wrapped_model_77568432

input_3596G
5model_3595_dense_17970_matmul_readvariableop_resource:(D
6model_3595_dense_17970_biasadd_readvariableop_resource:G
5model_3595_dense_17971_matmul_readvariableop_resource:(D
6model_3595_dense_17971_biasadd_readvariableop_resource:(G
5model_3595_dense_17972_matmul_readvariableop_resource:(
D
6model_3595_dense_17972_biasadd_readvariableop_resource:
G
5model_3595_dense_17973_matmul_readvariableop_resource:
(D
6model_3595_dense_17973_biasadd_readvariableop_resource:(G
5model_3595_dense_17974_matmul_readvariableop_resource:(D
6model_3595_dense_17974_biasadd_readvariableop_resource:G
5model_3595_dense_17975_matmul_readvariableop_resource:(D
6model_3595_dense_17975_biasadd_readvariableop_resource:(G
5model_3595_dense_17976_matmul_readvariableop_resource:(
D
6model_3595_dense_17976_biasadd_readvariableop_resource:
G
5model_3595_dense_17977_matmul_readvariableop_resource:
(D
6model_3595_dense_17977_biasadd_readvariableop_resource:(G
5model_3595_dense_17978_matmul_readvariableop_resource:(D
6model_3595_dense_17978_biasadd_readvariableop_resource:G
5model_3595_dense_17979_matmul_readvariableop_resource:(D
6model_3595_dense_17979_biasadd_readvariableop_resource:(
identity��-model_3595/dense_17970/BiasAdd/ReadVariableOp�,model_3595/dense_17970/MatMul/ReadVariableOp�-model_3595/dense_17971/BiasAdd/ReadVariableOp�,model_3595/dense_17971/MatMul/ReadVariableOp�-model_3595/dense_17972/BiasAdd/ReadVariableOp�,model_3595/dense_17972/MatMul/ReadVariableOp�-model_3595/dense_17973/BiasAdd/ReadVariableOp�,model_3595/dense_17973/MatMul/ReadVariableOp�-model_3595/dense_17974/BiasAdd/ReadVariableOp�,model_3595/dense_17974/MatMul/ReadVariableOp�-model_3595/dense_17975/BiasAdd/ReadVariableOp�,model_3595/dense_17975/MatMul/ReadVariableOp�-model_3595/dense_17976/BiasAdd/ReadVariableOp�,model_3595/dense_17976/MatMul/ReadVariableOp�-model_3595/dense_17977/BiasAdd/ReadVariableOp�,model_3595/dense_17977/MatMul/ReadVariableOp�-model_3595/dense_17978/BiasAdd/ReadVariableOp�,model_3595/dense_17978/MatMul/ReadVariableOp�-model_3595/dense_17979/BiasAdd/ReadVariableOp�,model_3595/dense_17979/MatMul/ReadVariableOp�
,model_3595/dense_17970/MatMul/ReadVariableOpReadVariableOp5model_3595_dense_17970_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
model_3595/dense_17970/MatMulMatMul
input_35964model_3595/dense_17970/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-model_3595/dense_17970/BiasAdd/ReadVariableOpReadVariableOp6model_3595_dense_17970_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_3595/dense_17970/BiasAddBiasAdd'model_3595/dense_17970/MatMul:product:05model_3595/dense_17970/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
model_3595/dense_17970/ReluRelu'model_3595/dense_17970/BiasAdd:output:0*
T0*'
_output_shapes
:����������
,model_3595/dense_17971/MatMul/ReadVariableOpReadVariableOp5model_3595_dense_17971_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
model_3595/dense_17971/MatMulMatMul)model_3595/dense_17970/Relu:activations:04model_3595/dense_17971/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
-model_3595/dense_17971/BiasAdd/ReadVariableOpReadVariableOp6model_3595_dense_17971_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
model_3595/dense_17971/BiasAddBiasAdd'model_3595/dense_17971/MatMul:product:05model_3595/dense_17971/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
,model_3595/dense_17972/MatMul/ReadVariableOpReadVariableOp5model_3595_dense_17972_matmul_readvariableop_resource*
_output_shapes

:(
*
dtype0�
model_3595/dense_17972/MatMulMatMul'model_3595/dense_17971/BiasAdd:output:04model_3595/dense_17972/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
-model_3595/dense_17972/BiasAdd/ReadVariableOpReadVariableOp6model_3595_dense_17972_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
model_3595/dense_17972/BiasAddBiasAdd'model_3595/dense_17972/MatMul:product:05model_3595/dense_17972/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
~
model_3595/dense_17972/ReluRelu'model_3595/dense_17972/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
,model_3595/dense_17973/MatMul/ReadVariableOpReadVariableOp5model_3595_dense_17973_matmul_readvariableop_resource*
_output_shapes

:
(*
dtype0�
model_3595/dense_17973/MatMulMatMul)model_3595/dense_17972/Relu:activations:04model_3595/dense_17973/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
-model_3595/dense_17973/BiasAdd/ReadVariableOpReadVariableOp6model_3595_dense_17973_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
model_3595/dense_17973/BiasAddBiasAdd'model_3595/dense_17973/MatMul:product:05model_3595/dense_17973/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
,model_3595/dense_17974/MatMul/ReadVariableOpReadVariableOp5model_3595_dense_17974_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
model_3595/dense_17974/MatMulMatMul'model_3595/dense_17973/BiasAdd:output:04model_3595/dense_17974/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-model_3595/dense_17974/BiasAdd/ReadVariableOpReadVariableOp6model_3595_dense_17974_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_3595/dense_17974/BiasAddBiasAdd'model_3595/dense_17974/MatMul:product:05model_3595/dense_17974/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
model_3595/dense_17974/ReluRelu'model_3595/dense_17974/BiasAdd:output:0*
T0*'
_output_shapes
:����������
,model_3595/dense_17975/MatMul/ReadVariableOpReadVariableOp5model_3595_dense_17975_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
model_3595/dense_17975/MatMulMatMul)model_3595/dense_17974/Relu:activations:04model_3595/dense_17975/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
-model_3595/dense_17975/BiasAdd/ReadVariableOpReadVariableOp6model_3595_dense_17975_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
model_3595/dense_17975/BiasAddBiasAdd'model_3595/dense_17975/MatMul:product:05model_3595/dense_17975/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
,model_3595/dense_17976/MatMul/ReadVariableOpReadVariableOp5model_3595_dense_17976_matmul_readvariableop_resource*
_output_shapes

:(
*
dtype0�
model_3595/dense_17976/MatMulMatMul'model_3595/dense_17975/BiasAdd:output:04model_3595/dense_17976/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
-model_3595/dense_17976/BiasAdd/ReadVariableOpReadVariableOp6model_3595_dense_17976_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
model_3595/dense_17976/BiasAddBiasAdd'model_3595/dense_17976/MatMul:product:05model_3595/dense_17976/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
~
model_3595/dense_17976/ReluRelu'model_3595/dense_17976/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
,model_3595/dense_17977/MatMul/ReadVariableOpReadVariableOp5model_3595_dense_17977_matmul_readvariableop_resource*
_output_shapes

:
(*
dtype0�
model_3595/dense_17977/MatMulMatMul)model_3595/dense_17976/Relu:activations:04model_3595/dense_17977/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
-model_3595/dense_17977/BiasAdd/ReadVariableOpReadVariableOp6model_3595_dense_17977_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
model_3595/dense_17977/BiasAddBiasAdd'model_3595/dense_17977/MatMul:product:05model_3595/dense_17977/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
,model_3595/dense_17978/MatMul/ReadVariableOpReadVariableOp5model_3595_dense_17978_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
model_3595/dense_17978/MatMulMatMul'model_3595/dense_17977/BiasAdd:output:04model_3595/dense_17978/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-model_3595/dense_17978/BiasAdd/ReadVariableOpReadVariableOp6model_3595_dense_17978_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_3595/dense_17978/BiasAddBiasAdd'model_3595/dense_17978/MatMul:product:05model_3595/dense_17978/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
model_3595/dense_17978/ReluRelu'model_3595/dense_17978/BiasAdd:output:0*
T0*'
_output_shapes
:����������
,model_3595/dense_17979/MatMul/ReadVariableOpReadVariableOp5model_3595_dense_17979_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
model_3595/dense_17979/MatMulMatMul)model_3595/dense_17978/Relu:activations:04model_3595/dense_17979/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
-model_3595/dense_17979/BiasAdd/ReadVariableOpReadVariableOp6model_3595_dense_17979_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
model_3595/dense_17979/BiasAddBiasAdd'model_3595/dense_17979/MatMul:product:05model_3595/dense_17979/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(v
IdentityIdentity'model_3595/dense_17979/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������(�
NoOpNoOp.^model_3595/dense_17970/BiasAdd/ReadVariableOp-^model_3595/dense_17970/MatMul/ReadVariableOp.^model_3595/dense_17971/BiasAdd/ReadVariableOp-^model_3595/dense_17971/MatMul/ReadVariableOp.^model_3595/dense_17972/BiasAdd/ReadVariableOp-^model_3595/dense_17972/MatMul/ReadVariableOp.^model_3595/dense_17973/BiasAdd/ReadVariableOp-^model_3595/dense_17973/MatMul/ReadVariableOp.^model_3595/dense_17974/BiasAdd/ReadVariableOp-^model_3595/dense_17974/MatMul/ReadVariableOp.^model_3595/dense_17975/BiasAdd/ReadVariableOp-^model_3595/dense_17975/MatMul/ReadVariableOp.^model_3595/dense_17976/BiasAdd/ReadVariableOp-^model_3595/dense_17976/MatMul/ReadVariableOp.^model_3595/dense_17977/BiasAdd/ReadVariableOp-^model_3595/dense_17977/MatMul/ReadVariableOp.^model_3595/dense_17978/BiasAdd/ReadVariableOp-^model_3595/dense_17978/MatMul/ReadVariableOp.^model_3595/dense_17979/BiasAdd/ReadVariableOp-^model_3595/dense_17979/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������(: : : : : : : : : : : : : : : : : : : : 2^
-model_3595/dense_17970/BiasAdd/ReadVariableOp-model_3595/dense_17970/BiasAdd/ReadVariableOp2\
,model_3595/dense_17970/MatMul/ReadVariableOp,model_3595/dense_17970/MatMul/ReadVariableOp2^
-model_3595/dense_17971/BiasAdd/ReadVariableOp-model_3595/dense_17971/BiasAdd/ReadVariableOp2\
,model_3595/dense_17971/MatMul/ReadVariableOp,model_3595/dense_17971/MatMul/ReadVariableOp2^
-model_3595/dense_17972/BiasAdd/ReadVariableOp-model_3595/dense_17972/BiasAdd/ReadVariableOp2\
,model_3595/dense_17972/MatMul/ReadVariableOp,model_3595/dense_17972/MatMul/ReadVariableOp2^
-model_3595/dense_17973/BiasAdd/ReadVariableOp-model_3595/dense_17973/BiasAdd/ReadVariableOp2\
,model_3595/dense_17973/MatMul/ReadVariableOp,model_3595/dense_17973/MatMul/ReadVariableOp2^
-model_3595/dense_17974/BiasAdd/ReadVariableOp-model_3595/dense_17974/BiasAdd/ReadVariableOp2\
,model_3595/dense_17974/MatMul/ReadVariableOp,model_3595/dense_17974/MatMul/ReadVariableOp2^
-model_3595/dense_17975/BiasAdd/ReadVariableOp-model_3595/dense_17975/BiasAdd/ReadVariableOp2\
,model_3595/dense_17975/MatMul/ReadVariableOp,model_3595/dense_17975/MatMul/ReadVariableOp2^
-model_3595/dense_17976/BiasAdd/ReadVariableOp-model_3595/dense_17976/BiasAdd/ReadVariableOp2\
,model_3595/dense_17976/MatMul/ReadVariableOp,model_3595/dense_17976/MatMul/ReadVariableOp2^
-model_3595/dense_17977/BiasAdd/ReadVariableOp-model_3595/dense_17977/BiasAdd/ReadVariableOp2\
,model_3595/dense_17977/MatMul/ReadVariableOp,model_3595/dense_17977/MatMul/ReadVariableOp2^
-model_3595/dense_17978/BiasAdd/ReadVariableOp-model_3595/dense_17978/BiasAdd/ReadVariableOp2\
,model_3595/dense_17978/MatMul/ReadVariableOp,model_3595/dense_17978/MatMul/ReadVariableOp2^
-model_3595/dense_17979/BiasAdd/ReadVariableOp-model_3595/dense_17979/BiasAdd/ReadVariableOp2\
,model_3595/dense_17979/MatMul/ReadVariableOp,model_3595/dense_17979/MatMul/ReadVariableOp:S O
'
_output_shapes
:���������(
$
_user_specified_name
input_3596
�	
�
I__inference_dense_17971_layer_call_and_return_conditional_losses_77569361

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
I__inference_dense_17977_layer_call_and_return_conditional_losses_77568562

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

�
I__inference_dense_17978_layer_call_and_return_conditional_losses_77569498

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
�7
�	
H__inference_model_3595_layer_call_and_return_conditional_losses_77568602

input_3596&
dense_17970_77568448:("
dense_17970_77568450:&
dense_17971_77568464:("
dense_17971_77568466:(&
dense_17972_77568481:(
"
dense_17972_77568483:
&
dense_17973_77568497:
("
dense_17973_77568499:(&
dense_17974_77568514:("
dense_17974_77568516:&
dense_17975_77568530:("
dense_17975_77568532:(&
dense_17976_77568547:(
"
dense_17976_77568549:
&
dense_17977_77568563:
("
dense_17977_77568565:(&
dense_17978_77568580:("
dense_17978_77568582:&
dense_17979_77568596:("
dense_17979_77568598:(
identity��#dense_17970/StatefulPartitionedCall�#dense_17971/StatefulPartitionedCall�#dense_17972/StatefulPartitionedCall�#dense_17973/StatefulPartitionedCall�#dense_17974/StatefulPartitionedCall�#dense_17975/StatefulPartitionedCall�#dense_17976/StatefulPartitionedCall�#dense_17977/StatefulPartitionedCall�#dense_17978/StatefulPartitionedCall�#dense_17979/StatefulPartitionedCall�
#dense_17970/StatefulPartitionedCallStatefulPartitionedCall
input_3596dense_17970_77568448dense_17970_77568450*
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
I__inference_dense_17970_layer_call_and_return_conditional_losses_77568447�
#dense_17971/StatefulPartitionedCallStatefulPartitionedCall,dense_17970/StatefulPartitionedCall:output:0dense_17971_77568464dense_17971_77568466*
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
I__inference_dense_17971_layer_call_and_return_conditional_losses_77568463�
#dense_17972/StatefulPartitionedCallStatefulPartitionedCall,dense_17971/StatefulPartitionedCall:output:0dense_17972_77568481dense_17972_77568483*
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
I__inference_dense_17972_layer_call_and_return_conditional_losses_77568480�
#dense_17973/StatefulPartitionedCallStatefulPartitionedCall,dense_17972/StatefulPartitionedCall:output:0dense_17973_77568497dense_17973_77568499*
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
I__inference_dense_17973_layer_call_and_return_conditional_losses_77568496�
#dense_17974/StatefulPartitionedCallStatefulPartitionedCall,dense_17973/StatefulPartitionedCall:output:0dense_17974_77568514dense_17974_77568516*
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
I__inference_dense_17974_layer_call_and_return_conditional_losses_77568513�
#dense_17975/StatefulPartitionedCallStatefulPartitionedCall,dense_17974/StatefulPartitionedCall:output:0dense_17975_77568530dense_17975_77568532*
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
I__inference_dense_17975_layer_call_and_return_conditional_losses_77568529�
#dense_17976/StatefulPartitionedCallStatefulPartitionedCall,dense_17975/StatefulPartitionedCall:output:0dense_17976_77568547dense_17976_77568549*
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
I__inference_dense_17976_layer_call_and_return_conditional_losses_77568546�
#dense_17977/StatefulPartitionedCallStatefulPartitionedCall,dense_17976/StatefulPartitionedCall:output:0dense_17977_77568563dense_17977_77568565*
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
I__inference_dense_17977_layer_call_and_return_conditional_losses_77568562�
#dense_17978/StatefulPartitionedCallStatefulPartitionedCall,dense_17977/StatefulPartitionedCall:output:0dense_17978_77568580dense_17978_77568582*
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
I__inference_dense_17978_layer_call_and_return_conditional_losses_77568579�
#dense_17979/StatefulPartitionedCallStatefulPartitionedCall,dense_17978/StatefulPartitionedCall:output:0dense_17979_77568596dense_17979_77568598*
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
I__inference_dense_17979_layer_call_and_return_conditional_losses_77568595{
IdentityIdentity,dense_17979/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(�
NoOpNoOp$^dense_17970/StatefulPartitionedCall$^dense_17971/StatefulPartitionedCall$^dense_17972/StatefulPartitionedCall$^dense_17973/StatefulPartitionedCall$^dense_17974/StatefulPartitionedCall$^dense_17975/StatefulPartitionedCall$^dense_17976/StatefulPartitionedCall$^dense_17977/StatefulPartitionedCall$^dense_17978/StatefulPartitionedCall$^dense_17979/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������(: : : : : : : : : : : : : : : : : : : : 2J
#dense_17970/StatefulPartitionedCall#dense_17970/StatefulPartitionedCall2J
#dense_17971/StatefulPartitionedCall#dense_17971/StatefulPartitionedCall2J
#dense_17972/StatefulPartitionedCall#dense_17972/StatefulPartitionedCall2J
#dense_17973/StatefulPartitionedCall#dense_17973/StatefulPartitionedCall2J
#dense_17974/StatefulPartitionedCall#dense_17974/StatefulPartitionedCall2J
#dense_17975/StatefulPartitionedCall#dense_17975/StatefulPartitionedCall2J
#dense_17976/StatefulPartitionedCall#dense_17976/StatefulPartitionedCall2J
#dense_17977/StatefulPartitionedCall#dense_17977/StatefulPartitionedCall2J
#dense_17978/StatefulPartitionedCall#dense_17978/StatefulPartitionedCall2J
#dense_17979/StatefulPartitionedCall#dense_17979/StatefulPartitionedCall:S O
'
_output_shapes
:���������(
$
_user_specified_name
input_3596
�7
�	
H__inference_model_3595_layer_call_and_return_conditional_losses_77568656

input_3596&
dense_17970_77568605:("
dense_17970_77568607:&
dense_17971_77568610:("
dense_17971_77568612:(&
dense_17972_77568615:(
"
dense_17972_77568617:
&
dense_17973_77568620:
("
dense_17973_77568622:(&
dense_17974_77568625:("
dense_17974_77568627:&
dense_17975_77568630:("
dense_17975_77568632:(&
dense_17976_77568635:(
"
dense_17976_77568637:
&
dense_17977_77568640:
("
dense_17977_77568642:(&
dense_17978_77568645:("
dense_17978_77568647:&
dense_17979_77568650:("
dense_17979_77568652:(
identity��#dense_17970/StatefulPartitionedCall�#dense_17971/StatefulPartitionedCall�#dense_17972/StatefulPartitionedCall�#dense_17973/StatefulPartitionedCall�#dense_17974/StatefulPartitionedCall�#dense_17975/StatefulPartitionedCall�#dense_17976/StatefulPartitionedCall�#dense_17977/StatefulPartitionedCall�#dense_17978/StatefulPartitionedCall�#dense_17979/StatefulPartitionedCall�
#dense_17970/StatefulPartitionedCallStatefulPartitionedCall
input_3596dense_17970_77568605dense_17970_77568607*
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
I__inference_dense_17970_layer_call_and_return_conditional_losses_77568447�
#dense_17971/StatefulPartitionedCallStatefulPartitionedCall,dense_17970/StatefulPartitionedCall:output:0dense_17971_77568610dense_17971_77568612*
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
I__inference_dense_17971_layer_call_and_return_conditional_losses_77568463�
#dense_17972/StatefulPartitionedCallStatefulPartitionedCall,dense_17971/StatefulPartitionedCall:output:0dense_17972_77568615dense_17972_77568617*
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
I__inference_dense_17972_layer_call_and_return_conditional_losses_77568480�
#dense_17973/StatefulPartitionedCallStatefulPartitionedCall,dense_17972/StatefulPartitionedCall:output:0dense_17973_77568620dense_17973_77568622*
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
I__inference_dense_17973_layer_call_and_return_conditional_losses_77568496�
#dense_17974/StatefulPartitionedCallStatefulPartitionedCall,dense_17973/StatefulPartitionedCall:output:0dense_17974_77568625dense_17974_77568627*
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
I__inference_dense_17974_layer_call_and_return_conditional_losses_77568513�
#dense_17975/StatefulPartitionedCallStatefulPartitionedCall,dense_17974/StatefulPartitionedCall:output:0dense_17975_77568630dense_17975_77568632*
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
I__inference_dense_17975_layer_call_and_return_conditional_losses_77568529�
#dense_17976/StatefulPartitionedCallStatefulPartitionedCall,dense_17975/StatefulPartitionedCall:output:0dense_17976_77568635dense_17976_77568637*
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
I__inference_dense_17976_layer_call_and_return_conditional_losses_77568546�
#dense_17977/StatefulPartitionedCallStatefulPartitionedCall,dense_17976/StatefulPartitionedCall:output:0dense_17977_77568640dense_17977_77568642*
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
I__inference_dense_17977_layer_call_and_return_conditional_losses_77568562�
#dense_17978/StatefulPartitionedCallStatefulPartitionedCall,dense_17977/StatefulPartitionedCall:output:0dense_17978_77568645dense_17978_77568647*
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
I__inference_dense_17978_layer_call_and_return_conditional_losses_77568579�
#dense_17979/StatefulPartitionedCallStatefulPartitionedCall,dense_17978/StatefulPartitionedCall:output:0dense_17979_77568650dense_17979_77568652*
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
I__inference_dense_17979_layer_call_and_return_conditional_losses_77568595{
IdentityIdentity,dense_17979/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(�
NoOpNoOp$^dense_17970/StatefulPartitionedCall$^dense_17971/StatefulPartitionedCall$^dense_17972/StatefulPartitionedCall$^dense_17973/StatefulPartitionedCall$^dense_17974/StatefulPartitionedCall$^dense_17975/StatefulPartitionedCall$^dense_17976/StatefulPartitionedCall$^dense_17977/StatefulPartitionedCall$^dense_17978/StatefulPartitionedCall$^dense_17979/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������(: : : : : : : : : : : : : : : : : : : : 2J
#dense_17970/StatefulPartitionedCall#dense_17970/StatefulPartitionedCall2J
#dense_17971/StatefulPartitionedCall#dense_17971/StatefulPartitionedCall2J
#dense_17972/StatefulPartitionedCall#dense_17972/StatefulPartitionedCall2J
#dense_17973/StatefulPartitionedCall#dense_17973/StatefulPartitionedCall2J
#dense_17974/StatefulPartitionedCall#dense_17974/StatefulPartitionedCall2J
#dense_17975/StatefulPartitionedCall#dense_17975/StatefulPartitionedCall2J
#dense_17976/StatefulPartitionedCall#dense_17976/StatefulPartitionedCall2J
#dense_17977/StatefulPartitionedCall#dense_17977/StatefulPartitionedCall2J
#dense_17978/StatefulPartitionedCall#dense_17978/StatefulPartitionedCall2J
#dense_17979/StatefulPartitionedCall#dense_17979/StatefulPartitionedCall:S O
'
_output_shapes
:���������(
$
_user_specified_name
input_3596
�

�
I__inference_dense_17978_layer_call_and_return_conditional_losses_77568579

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
I__inference_dense_17970_layer_call_and_return_conditional_losses_77568447

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
-__inference_model_3595_layer_call_fn_77568756

input_3596
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
input_3596unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
H__inference_model_3595_layer_call_and_return_conditional_losses_77568713o
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
input_3596
�7
�	
H__inference_model_3595_layer_call_and_return_conditional_losses_77568713

inputs&
dense_17970_77568662:("
dense_17970_77568664:&
dense_17971_77568667:("
dense_17971_77568669:(&
dense_17972_77568672:(
"
dense_17972_77568674:
&
dense_17973_77568677:
("
dense_17973_77568679:(&
dense_17974_77568682:("
dense_17974_77568684:&
dense_17975_77568687:("
dense_17975_77568689:(&
dense_17976_77568692:(
"
dense_17976_77568694:
&
dense_17977_77568697:
("
dense_17977_77568699:(&
dense_17978_77568702:("
dense_17978_77568704:&
dense_17979_77568707:("
dense_17979_77568709:(
identity��#dense_17970/StatefulPartitionedCall�#dense_17971/StatefulPartitionedCall�#dense_17972/StatefulPartitionedCall�#dense_17973/StatefulPartitionedCall�#dense_17974/StatefulPartitionedCall�#dense_17975/StatefulPartitionedCall�#dense_17976/StatefulPartitionedCall�#dense_17977/StatefulPartitionedCall�#dense_17978/StatefulPartitionedCall�#dense_17979/StatefulPartitionedCall�
#dense_17970/StatefulPartitionedCallStatefulPartitionedCallinputsdense_17970_77568662dense_17970_77568664*
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
I__inference_dense_17970_layer_call_and_return_conditional_losses_77568447�
#dense_17971/StatefulPartitionedCallStatefulPartitionedCall,dense_17970/StatefulPartitionedCall:output:0dense_17971_77568667dense_17971_77568669*
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
I__inference_dense_17971_layer_call_and_return_conditional_losses_77568463�
#dense_17972/StatefulPartitionedCallStatefulPartitionedCall,dense_17971/StatefulPartitionedCall:output:0dense_17972_77568672dense_17972_77568674*
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
I__inference_dense_17972_layer_call_and_return_conditional_losses_77568480�
#dense_17973/StatefulPartitionedCallStatefulPartitionedCall,dense_17972/StatefulPartitionedCall:output:0dense_17973_77568677dense_17973_77568679*
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
I__inference_dense_17973_layer_call_and_return_conditional_losses_77568496�
#dense_17974/StatefulPartitionedCallStatefulPartitionedCall,dense_17973/StatefulPartitionedCall:output:0dense_17974_77568682dense_17974_77568684*
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
I__inference_dense_17974_layer_call_and_return_conditional_losses_77568513�
#dense_17975/StatefulPartitionedCallStatefulPartitionedCall,dense_17974/StatefulPartitionedCall:output:0dense_17975_77568687dense_17975_77568689*
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
I__inference_dense_17975_layer_call_and_return_conditional_losses_77568529�
#dense_17976/StatefulPartitionedCallStatefulPartitionedCall,dense_17975/StatefulPartitionedCall:output:0dense_17976_77568692dense_17976_77568694*
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
I__inference_dense_17976_layer_call_and_return_conditional_losses_77568546�
#dense_17977/StatefulPartitionedCallStatefulPartitionedCall,dense_17976/StatefulPartitionedCall:output:0dense_17977_77568697dense_17977_77568699*
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
I__inference_dense_17977_layer_call_and_return_conditional_losses_77568562�
#dense_17978/StatefulPartitionedCallStatefulPartitionedCall,dense_17977/StatefulPartitionedCall:output:0dense_17978_77568702dense_17978_77568704*
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
I__inference_dense_17978_layer_call_and_return_conditional_losses_77568579�
#dense_17979/StatefulPartitionedCallStatefulPartitionedCall,dense_17978/StatefulPartitionedCall:output:0dense_17979_77568707dense_17979_77568709*
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
I__inference_dense_17979_layer_call_and_return_conditional_losses_77568595{
IdentityIdentity,dense_17979/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(�
NoOpNoOp$^dense_17970/StatefulPartitionedCall$^dense_17971/StatefulPartitionedCall$^dense_17972/StatefulPartitionedCall$^dense_17973/StatefulPartitionedCall$^dense_17974/StatefulPartitionedCall$^dense_17975/StatefulPartitionedCall$^dense_17976/StatefulPartitionedCall$^dense_17977/StatefulPartitionedCall$^dense_17978/StatefulPartitionedCall$^dense_17979/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������(: : : : : : : : : : : : : : : : : : : : 2J
#dense_17970/StatefulPartitionedCall#dense_17970/StatefulPartitionedCall2J
#dense_17971/StatefulPartitionedCall#dense_17971/StatefulPartitionedCall2J
#dense_17972/StatefulPartitionedCall#dense_17972/StatefulPartitionedCall2J
#dense_17973/StatefulPartitionedCall#dense_17973/StatefulPartitionedCall2J
#dense_17974/StatefulPartitionedCall#dense_17974/StatefulPartitionedCall2J
#dense_17975/StatefulPartitionedCall#dense_17975/StatefulPartitionedCall2J
#dense_17976/StatefulPartitionedCall#dense_17976/StatefulPartitionedCall2J
#dense_17977/StatefulPartitionedCall#dense_17977/StatefulPartitionedCall2J
#dense_17978/StatefulPartitionedCall#dense_17978/StatefulPartitionedCall2J
#dense_17979/StatefulPartitionedCall#dense_17979/StatefulPartitionedCall:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�7
�	
H__inference_model_3595_layer_call_and_return_conditional_losses_77568812

inputs&
dense_17970_77568761:("
dense_17970_77568763:&
dense_17971_77568766:("
dense_17971_77568768:(&
dense_17972_77568771:(
"
dense_17972_77568773:
&
dense_17973_77568776:
("
dense_17973_77568778:(&
dense_17974_77568781:("
dense_17974_77568783:&
dense_17975_77568786:("
dense_17975_77568788:(&
dense_17976_77568791:(
"
dense_17976_77568793:
&
dense_17977_77568796:
("
dense_17977_77568798:(&
dense_17978_77568801:("
dense_17978_77568803:&
dense_17979_77568806:("
dense_17979_77568808:(
identity��#dense_17970/StatefulPartitionedCall�#dense_17971/StatefulPartitionedCall�#dense_17972/StatefulPartitionedCall�#dense_17973/StatefulPartitionedCall�#dense_17974/StatefulPartitionedCall�#dense_17975/StatefulPartitionedCall�#dense_17976/StatefulPartitionedCall�#dense_17977/StatefulPartitionedCall�#dense_17978/StatefulPartitionedCall�#dense_17979/StatefulPartitionedCall�
#dense_17970/StatefulPartitionedCallStatefulPartitionedCallinputsdense_17970_77568761dense_17970_77568763*
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
I__inference_dense_17970_layer_call_and_return_conditional_losses_77568447�
#dense_17971/StatefulPartitionedCallStatefulPartitionedCall,dense_17970/StatefulPartitionedCall:output:0dense_17971_77568766dense_17971_77568768*
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
I__inference_dense_17971_layer_call_and_return_conditional_losses_77568463�
#dense_17972/StatefulPartitionedCallStatefulPartitionedCall,dense_17971/StatefulPartitionedCall:output:0dense_17972_77568771dense_17972_77568773*
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
I__inference_dense_17972_layer_call_and_return_conditional_losses_77568480�
#dense_17973/StatefulPartitionedCallStatefulPartitionedCall,dense_17972/StatefulPartitionedCall:output:0dense_17973_77568776dense_17973_77568778*
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
I__inference_dense_17973_layer_call_and_return_conditional_losses_77568496�
#dense_17974/StatefulPartitionedCallStatefulPartitionedCall,dense_17973/StatefulPartitionedCall:output:0dense_17974_77568781dense_17974_77568783*
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
I__inference_dense_17974_layer_call_and_return_conditional_losses_77568513�
#dense_17975/StatefulPartitionedCallStatefulPartitionedCall,dense_17974/StatefulPartitionedCall:output:0dense_17975_77568786dense_17975_77568788*
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
I__inference_dense_17975_layer_call_and_return_conditional_losses_77568529�
#dense_17976/StatefulPartitionedCallStatefulPartitionedCall,dense_17975/StatefulPartitionedCall:output:0dense_17976_77568791dense_17976_77568793*
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
I__inference_dense_17976_layer_call_and_return_conditional_losses_77568546�
#dense_17977/StatefulPartitionedCallStatefulPartitionedCall,dense_17976/StatefulPartitionedCall:output:0dense_17977_77568796dense_17977_77568798*
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
I__inference_dense_17977_layer_call_and_return_conditional_losses_77568562�
#dense_17978/StatefulPartitionedCallStatefulPartitionedCall,dense_17977/StatefulPartitionedCall:output:0dense_17978_77568801dense_17978_77568803*
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
I__inference_dense_17978_layer_call_and_return_conditional_losses_77568579�
#dense_17979/StatefulPartitionedCallStatefulPartitionedCall,dense_17978/StatefulPartitionedCall:output:0dense_17979_77568806dense_17979_77568808*
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
I__inference_dense_17979_layer_call_and_return_conditional_losses_77568595{
IdentityIdentity,dense_17979/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(�
NoOpNoOp$^dense_17970/StatefulPartitionedCall$^dense_17971/StatefulPartitionedCall$^dense_17972/StatefulPartitionedCall$^dense_17973/StatefulPartitionedCall$^dense_17974/StatefulPartitionedCall$^dense_17975/StatefulPartitionedCall$^dense_17976/StatefulPartitionedCall$^dense_17977/StatefulPartitionedCall$^dense_17978/StatefulPartitionedCall$^dense_17979/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������(: : : : : : : : : : : : : : : : : : : : 2J
#dense_17970/StatefulPartitionedCall#dense_17970/StatefulPartitionedCall2J
#dense_17971/StatefulPartitionedCall#dense_17971/StatefulPartitionedCall2J
#dense_17972/StatefulPartitionedCall#dense_17972/StatefulPartitionedCall2J
#dense_17973/StatefulPartitionedCall#dense_17973/StatefulPartitionedCall2J
#dense_17974/StatefulPartitionedCall#dense_17974/StatefulPartitionedCall2J
#dense_17975/StatefulPartitionedCall#dense_17975/StatefulPartitionedCall2J
#dense_17976/StatefulPartitionedCall#dense_17976/StatefulPartitionedCall2J
#dense_17977/StatefulPartitionedCall#dense_17977/StatefulPartitionedCall2J
#dense_17978/StatefulPartitionedCall#dense_17978/StatefulPartitionedCall2J
#dense_17979/StatefulPartitionedCall#dense_17979/StatefulPartitionedCall:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�
�
.__inference_dense_17978_layer_call_fn_77569487

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
I__inference_dense_17978_layer_call_and_return_conditional_losses_77568579o
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
�	
�
I__inference_dense_17973_layer_call_and_return_conditional_losses_77568496

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
�V
�
H__inference_model_3595_layer_call_and_return_conditional_losses_77569253

inputs<
*dense_17970_matmul_readvariableop_resource:(9
+dense_17970_biasadd_readvariableop_resource:<
*dense_17971_matmul_readvariableop_resource:(9
+dense_17971_biasadd_readvariableop_resource:(<
*dense_17972_matmul_readvariableop_resource:(
9
+dense_17972_biasadd_readvariableop_resource:
<
*dense_17973_matmul_readvariableop_resource:
(9
+dense_17973_biasadd_readvariableop_resource:(<
*dense_17974_matmul_readvariableop_resource:(9
+dense_17974_biasadd_readvariableop_resource:<
*dense_17975_matmul_readvariableop_resource:(9
+dense_17975_biasadd_readvariableop_resource:(<
*dense_17976_matmul_readvariableop_resource:(
9
+dense_17976_biasadd_readvariableop_resource:
<
*dense_17977_matmul_readvariableop_resource:
(9
+dense_17977_biasadd_readvariableop_resource:(<
*dense_17978_matmul_readvariableop_resource:(9
+dense_17978_biasadd_readvariableop_resource:<
*dense_17979_matmul_readvariableop_resource:(9
+dense_17979_biasadd_readvariableop_resource:(
identity��"dense_17970/BiasAdd/ReadVariableOp�!dense_17970/MatMul/ReadVariableOp�"dense_17971/BiasAdd/ReadVariableOp�!dense_17971/MatMul/ReadVariableOp�"dense_17972/BiasAdd/ReadVariableOp�!dense_17972/MatMul/ReadVariableOp�"dense_17973/BiasAdd/ReadVariableOp�!dense_17973/MatMul/ReadVariableOp�"dense_17974/BiasAdd/ReadVariableOp�!dense_17974/MatMul/ReadVariableOp�"dense_17975/BiasAdd/ReadVariableOp�!dense_17975/MatMul/ReadVariableOp�"dense_17976/BiasAdd/ReadVariableOp�!dense_17976/MatMul/ReadVariableOp�"dense_17977/BiasAdd/ReadVariableOp�!dense_17977/MatMul/ReadVariableOp�"dense_17978/BiasAdd/ReadVariableOp�!dense_17978/MatMul/ReadVariableOp�"dense_17979/BiasAdd/ReadVariableOp�!dense_17979/MatMul/ReadVariableOp�
!dense_17970/MatMul/ReadVariableOpReadVariableOp*dense_17970_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
dense_17970/MatMulMatMulinputs)dense_17970/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"dense_17970/BiasAdd/ReadVariableOpReadVariableOp+dense_17970_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_17970/BiasAddBiasAdddense_17970/MatMul:product:0*dense_17970/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
dense_17970/ReluReludense_17970/BiasAdd:output:0*
T0*'
_output_shapes
:����������
!dense_17971/MatMul/ReadVariableOpReadVariableOp*dense_17971_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
dense_17971/MatMulMatMuldense_17970/Relu:activations:0)dense_17971/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
"dense_17971/BiasAdd/ReadVariableOpReadVariableOp+dense_17971_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
dense_17971/BiasAddBiasAdddense_17971/MatMul:product:0*dense_17971/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
!dense_17972/MatMul/ReadVariableOpReadVariableOp*dense_17972_matmul_readvariableop_resource*
_output_shapes

:(
*
dtype0�
dense_17972/MatMulMatMuldense_17971/BiasAdd:output:0)dense_17972/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
"dense_17972/BiasAdd/ReadVariableOpReadVariableOp+dense_17972_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
dense_17972/BiasAddBiasAdddense_17972/MatMul:product:0*dense_17972/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
h
dense_17972/ReluReludense_17972/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
!dense_17973/MatMul/ReadVariableOpReadVariableOp*dense_17973_matmul_readvariableop_resource*
_output_shapes

:
(*
dtype0�
dense_17973/MatMulMatMuldense_17972/Relu:activations:0)dense_17973/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
"dense_17973/BiasAdd/ReadVariableOpReadVariableOp+dense_17973_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
dense_17973/BiasAddBiasAdddense_17973/MatMul:product:0*dense_17973/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
!dense_17974/MatMul/ReadVariableOpReadVariableOp*dense_17974_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
dense_17974/MatMulMatMuldense_17973/BiasAdd:output:0)dense_17974/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"dense_17974/BiasAdd/ReadVariableOpReadVariableOp+dense_17974_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_17974/BiasAddBiasAdddense_17974/MatMul:product:0*dense_17974/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
dense_17974/ReluReludense_17974/BiasAdd:output:0*
T0*'
_output_shapes
:����������
!dense_17975/MatMul/ReadVariableOpReadVariableOp*dense_17975_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
dense_17975/MatMulMatMuldense_17974/Relu:activations:0)dense_17975/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
"dense_17975/BiasAdd/ReadVariableOpReadVariableOp+dense_17975_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
dense_17975/BiasAddBiasAdddense_17975/MatMul:product:0*dense_17975/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
!dense_17976/MatMul/ReadVariableOpReadVariableOp*dense_17976_matmul_readvariableop_resource*
_output_shapes

:(
*
dtype0�
dense_17976/MatMulMatMuldense_17975/BiasAdd:output:0)dense_17976/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
"dense_17976/BiasAdd/ReadVariableOpReadVariableOp+dense_17976_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
dense_17976/BiasAddBiasAdddense_17976/MatMul:product:0*dense_17976/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
h
dense_17976/ReluReludense_17976/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
!dense_17977/MatMul/ReadVariableOpReadVariableOp*dense_17977_matmul_readvariableop_resource*
_output_shapes

:
(*
dtype0�
dense_17977/MatMulMatMuldense_17976/Relu:activations:0)dense_17977/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
"dense_17977/BiasAdd/ReadVariableOpReadVariableOp+dense_17977_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
dense_17977/BiasAddBiasAdddense_17977/MatMul:product:0*dense_17977/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
!dense_17978/MatMul/ReadVariableOpReadVariableOp*dense_17978_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
dense_17978/MatMulMatMuldense_17977/BiasAdd:output:0)dense_17978/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"dense_17978/BiasAdd/ReadVariableOpReadVariableOp+dense_17978_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_17978/BiasAddBiasAdddense_17978/MatMul:product:0*dense_17978/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
dense_17978/ReluReludense_17978/BiasAdd:output:0*
T0*'
_output_shapes
:����������
!dense_17979/MatMul/ReadVariableOpReadVariableOp*dense_17979_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
dense_17979/MatMulMatMuldense_17978/Relu:activations:0)dense_17979/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
"dense_17979/BiasAdd/ReadVariableOpReadVariableOp+dense_17979_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
dense_17979/BiasAddBiasAdddense_17979/MatMul:product:0*dense_17979/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(k
IdentityIdentitydense_17979/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������(�
NoOpNoOp#^dense_17970/BiasAdd/ReadVariableOp"^dense_17970/MatMul/ReadVariableOp#^dense_17971/BiasAdd/ReadVariableOp"^dense_17971/MatMul/ReadVariableOp#^dense_17972/BiasAdd/ReadVariableOp"^dense_17972/MatMul/ReadVariableOp#^dense_17973/BiasAdd/ReadVariableOp"^dense_17973/MatMul/ReadVariableOp#^dense_17974/BiasAdd/ReadVariableOp"^dense_17974/MatMul/ReadVariableOp#^dense_17975/BiasAdd/ReadVariableOp"^dense_17975/MatMul/ReadVariableOp#^dense_17976/BiasAdd/ReadVariableOp"^dense_17976/MatMul/ReadVariableOp#^dense_17977/BiasAdd/ReadVariableOp"^dense_17977/MatMul/ReadVariableOp#^dense_17978/BiasAdd/ReadVariableOp"^dense_17978/MatMul/ReadVariableOp#^dense_17979/BiasAdd/ReadVariableOp"^dense_17979/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������(: : : : : : : : : : : : : : : : : : : : 2H
"dense_17970/BiasAdd/ReadVariableOp"dense_17970/BiasAdd/ReadVariableOp2F
!dense_17970/MatMul/ReadVariableOp!dense_17970/MatMul/ReadVariableOp2H
"dense_17971/BiasAdd/ReadVariableOp"dense_17971/BiasAdd/ReadVariableOp2F
!dense_17971/MatMul/ReadVariableOp!dense_17971/MatMul/ReadVariableOp2H
"dense_17972/BiasAdd/ReadVariableOp"dense_17972/BiasAdd/ReadVariableOp2F
!dense_17972/MatMul/ReadVariableOp!dense_17972/MatMul/ReadVariableOp2H
"dense_17973/BiasAdd/ReadVariableOp"dense_17973/BiasAdd/ReadVariableOp2F
!dense_17973/MatMul/ReadVariableOp!dense_17973/MatMul/ReadVariableOp2H
"dense_17974/BiasAdd/ReadVariableOp"dense_17974/BiasAdd/ReadVariableOp2F
!dense_17974/MatMul/ReadVariableOp!dense_17974/MatMul/ReadVariableOp2H
"dense_17975/BiasAdd/ReadVariableOp"dense_17975/BiasAdd/ReadVariableOp2F
!dense_17975/MatMul/ReadVariableOp!dense_17975/MatMul/ReadVariableOp2H
"dense_17976/BiasAdd/ReadVariableOp"dense_17976/BiasAdd/ReadVariableOp2F
!dense_17976/MatMul/ReadVariableOp!dense_17976/MatMul/ReadVariableOp2H
"dense_17977/BiasAdd/ReadVariableOp"dense_17977/BiasAdd/ReadVariableOp2F
!dense_17977/MatMul/ReadVariableOp!dense_17977/MatMul/ReadVariableOp2H
"dense_17978/BiasAdd/ReadVariableOp"dense_17978/BiasAdd/ReadVariableOp2F
!dense_17978/MatMul/ReadVariableOp!dense_17978/MatMul/ReadVariableOp2H
"dense_17979/BiasAdd/ReadVariableOp"dense_17979/BiasAdd/ReadVariableOp2F
!dense_17979/MatMul/ReadVariableOp!dense_17979/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�
�
.__inference_dense_17972_layer_call_fn_77569370

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
I__inference_dense_17972_layer_call_and_return_conditional_losses_77568480o
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
�V
�
H__inference_model_3595_layer_call_and_return_conditional_losses_77569322

inputs<
*dense_17970_matmul_readvariableop_resource:(9
+dense_17970_biasadd_readvariableop_resource:<
*dense_17971_matmul_readvariableop_resource:(9
+dense_17971_biasadd_readvariableop_resource:(<
*dense_17972_matmul_readvariableop_resource:(
9
+dense_17972_biasadd_readvariableop_resource:
<
*dense_17973_matmul_readvariableop_resource:
(9
+dense_17973_biasadd_readvariableop_resource:(<
*dense_17974_matmul_readvariableop_resource:(9
+dense_17974_biasadd_readvariableop_resource:<
*dense_17975_matmul_readvariableop_resource:(9
+dense_17975_biasadd_readvariableop_resource:(<
*dense_17976_matmul_readvariableop_resource:(
9
+dense_17976_biasadd_readvariableop_resource:
<
*dense_17977_matmul_readvariableop_resource:
(9
+dense_17977_biasadd_readvariableop_resource:(<
*dense_17978_matmul_readvariableop_resource:(9
+dense_17978_biasadd_readvariableop_resource:<
*dense_17979_matmul_readvariableop_resource:(9
+dense_17979_biasadd_readvariableop_resource:(
identity��"dense_17970/BiasAdd/ReadVariableOp�!dense_17970/MatMul/ReadVariableOp�"dense_17971/BiasAdd/ReadVariableOp�!dense_17971/MatMul/ReadVariableOp�"dense_17972/BiasAdd/ReadVariableOp�!dense_17972/MatMul/ReadVariableOp�"dense_17973/BiasAdd/ReadVariableOp�!dense_17973/MatMul/ReadVariableOp�"dense_17974/BiasAdd/ReadVariableOp�!dense_17974/MatMul/ReadVariableOp�"dense_17975/BiasAdd/ReadVariableOp�!dense_17975/MatMul/ReadVariableOp�"dense_17976/BiasAdd/ReadVariableOp�!dense_17976/MatMul/ReadVariableOp�"dense_17977/BiasAdd/ReadVariableOp�!dense_17977/MatMul/ReadVariableOp�"dense_17978/BiasAdd/ReadVariableOp�!dense_17978/MatMul/ReadVariableOp�"dense_17979/BiasAdd/ReadVariableOp�!dense_17979/MatMul/ReadVariableOp�
!dense_17970/MatMul/ReadVariableOpReadVariableOp*dense_17970_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
dense_17970/MatMulMatMulinputs)dense_17970/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"dense_17970/BiasAdd/ReadVariableOpReadVariableOp+dense_17970_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_17970/BiasAddBiasAdddense_17970/MatMul:product:0*dense_17970/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
dense_17970/ReluReludense_17970/BiasAdd:output:0*
T0*'
_output_shapes
:����������
!dense_17971/MatMul/ReadVariableOpReadVariableOp*dense_17971_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
dense_17971/MatMulMatMuldense_17970/Relu:activations:0)dense_17971/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
"dense_17971/BiasAdd/ReadVariableOpReadVariableOp+dense_17971_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
dense_17971/BiasAddBiasAdddense_17971/MatMul:product:0*dense_17971/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
!dense_17972/MatMul/ReadVariableOpReadVariableOp*dense_17972_matmul_readvariableop_resource*
_output_shapes

:(
*
dtype0�
dense_17972/MatMulMatMuldense_17971/BiasAdd:output:0)dense_17972/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
"dense_17972/BiasAdd/ReadVariableOpReadVariableOp+dense_17972_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
dense_17972/BiasAddBiasAdddense_17972/MatMul:product:0*dense_17972/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
h
dense_17972/ReluReludense_17972/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
!dense_17973/MatMul/ReadVariableOpReadVariableOp*dense_17973_matmul_readvariableop_resource*
_output_shapes

:
(*
dtype0�
dense_17973/MatMulMatMuldense_17972/Relu:activations:0)dense_17973/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
"dense_17973/BiasAdd/ReadVariableOpReadVariableOp+dense_17973_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
dense_17973/BiasAddBiasAdddense_17973/MatMul:product:0*dense_17973/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
!dense_17974/MatMul/ReadVariableOpReadVariableOp*dense_17974_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
dense_17974/MatMulMatMuldense_17973/BiasAdd:output:0)dense_17974/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"dense_17974/BiasAdd/ReadVariableOpReadVariableOp+dense_17974_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_17974/BiasAddBiasAdddense_17974/MatMul:product:0*dense_17974/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
dense_17974/ReluReludense_17974/BiasAdd:output:0*
T0*'
_output_shapes
:����������
!dense_17975/MatMul/ReadVariableOpReadVariableOp*dense_17975_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
dense_17975/MatMulMatMuldense_17974/Relu:activations:0)dense_17975/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
"dense_17975/BiasAdd/ReadVariableOpReadVariableOp+dense_17975_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
dense_17975/BiasAddBiasAdddense_17975/MatMul:product:0*dense_17975/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
!dense_17976/MatMul/ReadVariableOpReadVariableOp*dense_17976_matmul_readvariableop_resource*
_output_shapes

:(
*
dtype0�
dense_17976/MatMulMatMuldense_17975/BiasAdd:output:0)dense_17976/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
"dense_17976/BiasAdd/ReadVariableOpReadVariableOp+dense_17976_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
dense_17976/BiasAddBiasAdddense_17976/MatMul:product:0*dense_17976/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
h
dense_17976/ReluReludense_17976/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
!dense_17977/MatMul/ReadVariableOpReadVariableOp*dense_17977_matmul_readvariableop_resource*
_output_shapes

:
(*
dtype0�
dense_17977/MatMulMatMuldense_17976/Relu:activations:0)dense_17977/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
"dense_17977/BiasAdd/ReadVariableOpReadVariableOp+dense_17977_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
dense_17977/BiasAddBiasAdddense_17977/MatMul:product:0*dense_17977/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
!dense_17978/MatMul/ReadVariableOpReadVariableOp*dense_17978_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
dense_17978/MatMulMatMuldense_17977/BiasAdd:output:0)dense_17978/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"dense_17978/BiasAdd/ReadVariableOpReadVariableOp+dense_17978_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_17978/BiasAddBiasAdddense_17978/MatMul:product:0*dense_17978/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
dense_17978/ReluReludense_17978/BiasAdd:output:0*
T0*'
_output_shapes
:����������
!dense_17979/MatMul/ReadVariableOpReadVariableOp*dense_17979_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0�
dense_17979/MatMulMatMuldense_17978/Relu:activations:0)dense_17979/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(�
"dense_17979/BiasAdd/ReadVariableOpReadVariableOp+dense_17979_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0�
dense_17979/BiasAddBiasAdddense_17979/MatMul:product:0*dense_17979/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(k
IdentityIdentitydense_17979/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������(�
NoOpNoOp#^dense_17970/BiasAdd/ReadVariableOp"^dense_17970/MatMul/ReadVariableOp#^dense_17971/BiasAdd/ReadVariableOp"^dense_17971/MatMul/ReadVariableOp#^dense_17972/BiasAdd/ReadVariableOp"^dense_17972/MatMul/ReadVariableOp#^dense_17973/BiasAdd/ReadVariableOp"^dense_17973/MatMul/ReadVariableOp#^dense_17974/BiasAdd/ReadVariableOp"^dense_17974/MatMul/ReadVariableOp#^dense_17975/BiasAdd/ReadVariableOp"^dense_17975/MatMul/ReadVariableOp#^dense_17976/BiasAdd/ReadVariableOp"^dense_17976/MatMul/ReadVariableOp#^dense_17977/BiasAdd/ReadVariableOp"^dense_17977/MatMul/ReadVariableOp#^dense_17978/BiasAdd/ReadVariableOp"^dense_17978/MatMul/ReadVariableOp#^dense_17979/BiasAdd/ReadVariableOp"^dense_17979/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������(: : : : : : : : : : : : : : : : : : : : 2H
"dense_17970/BiasAdd/ReadVariableOp"dense_17970/BiasAdd/ReadVariableOp2F
!dense_17970/MatMul/ReadVariableOp!dense_17970/MatMul/ReadVariableOp2H
"dense_17971/BiasAdd/ReadVariableOp"dense_17971/BiasAdd/ReadVariableOp2F
!dense_17971/MatMul/ReadVariableOp!dense_17971/MatMul/ReadVariableOp2H
"dense_17972/BiasAdd/ReadVariableOp"dense_17972/BiasAdd/ReadVariableOp2F
!dense_17972/MatMul/ReadVariableOp!dense_17972/MatMul/ReadVariableOp2H
"dense_17973/BiasAdd/ReadVariableOp"dense_17973/BiasAdd/ReadVariableOp2F
!dense_17973/MatMul/ReadVariableOp!dense_17973/MatMul/ReadVariableOp2H
"dense_17974/BiasAdd/ReadVariableOp"dense_17974/BiasAdd/ReadVariableOp2F
!dense_17974/MatMul/ReadVariableOp!dense_17974/MatMul/ReadVariableOp2H
"dense_17975/BiasAdd/ReadVariableOp"dense_17975/BiasAdd/ReadVariableOp2F
!dense_17975/MatMul/ReadVariableOp!dense_17975/MatMul/ReadVariableOp2H
"dense_17976/BiasAdd/ReadVariableOp"dense_17976/BiasAdd/ReadVariableOp2F
!dense_17976/MatMul/ReadVariableOp!dense_17976/MatMul/ReadVariableOp2H
"dense_17977/BiasAdd/ReadVariableOp"dense_17977/BiasAdd/ReadVariableOp2F
!dense_17977/MatMul/ReadVariableOp!dense_17977/MatMul/ReadVariableOp2H
"dense_17978/BiasAdd/ReadVariableOp"dense_17978/BiasAdd/ReadVariableOp2F
!dense_17978/MatMul/ReadVariableOp!dense_17978/MatMul/ReadVariableOp2H
"dense_17979/BiasAdd/ReadVariableOp"dense_17979/BiasAdd/ReadVariableOp2F
!dense_17979/MatMul/ReadVariableOp!dense_17979/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�
�
-__inference_model_3595_layer_call_fn_77569184

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
H__inference_model_3595_layer_call_and_return_conditional_losses_77568812o
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
�
�
.__inference_dense_17973_layer_call_fn_77569390

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
I__inference_dense_17973_layer_call_and_return_conditional_losses_77568496o
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

 
_user_specified_nameinputs"�
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

input_35963
serving_default_input_3596:0���������(?
dense_179790
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
-__inference_model_3595_layer_call_fn_77568756
-__inference_model_3595_layer_call_fn_77568855
-__inference_model_3595_layer_call_fn_77569139
-__inference_model_3595_layer_call_fn_77569184�
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
H__inference_model_3595_layer_call_and_return_conditional_losses_77568602
H__inference_model_3595_layer_call_and_return_conditional_losses_77568656
H__inference_model_3595_layer_call_and_return_conditional_losses_77569253
H__inference_model_3595_layer_call_and_return_conditional_losses_77569322�
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
#__inference__wrapped_model_77568432
input_3596"�
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
.__inference_dense_17970_layer_call_fn_77569331�
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
I__inference_dense_17970_layer_call_and_return_conditional_losses_77569342�
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
$:"(2dense_17970/kernel
:2dense_17970/bias
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
.__inference_dense_17971_layer_call_fn_77569351�
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
I__inference_dense_17971_layer_call_and_return_conditional_losses_77569361�
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
$:"(2dense_17971/kernel
:(2dense_17971/bias
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
.__inference_dense_17972_layer_call_fn_77569370�
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
I__inference_dense_17972_layer_call_and_return_conditional_losses_77569381�
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
2dense_17972/kernel
:
2dense_17972/bias
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
.__inference_dense_17973_layer_call_fn_77569390�
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
I__inference_dense_17973_layer_call_and_return_conditional_losses_77569400�
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
(2dense_17973/kernel
:(2dense_17973/bias
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
.__inference_dense_17974_layer_call_fn_77569409�
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
I__inference_dense_17974_layer_call_and_return_conditional_losses_77569420�
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
$:"(2dense_17974/kernel
:2dense_17974/bias
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
.__inference_dense_17975_layer_call_fn_77569429�
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
I__inference_dense_17975_layer_call_and_return_conditional_losses_77569439�
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
$:"(2dense_17975/kernel
:(2dense_17975/bias
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
.__inference_dense_17976_layer_call_fn_77569448�
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
I__inference_dense_17976_layer_call_and_return_conditional_losses_77569459�
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
2dense_17976/kernel
:
2dense_17976/bias
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
.__inference_dense_17977_layer_call_fn_77569468�
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
I__inference_dense_17977_layer_call_and_return_conditional_losses_77569478�
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
(2dense_17977/kernel
:(2dense_17977/bias
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
.__inference_dense_17978_layer_call_fn_77569487�
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
I__inference_dense_17978_layer_call_and_return_conditional_losses_77569498�
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
$:"(2dense_17978/kernel
:2dense_17978/bias
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
.__inference_dense_17979_layer_call_fn_77569507�
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
I__inference_dense_17979_layer_call_and_return_conditional_losses_77569517�
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
$:"(2dense_17979/kernel
:(2dense_17979/bias
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
-__inference_model_3595_layer_call_fn_77568756
input_3596"�
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
-__inference_model_3595_layer_call_fn_77568855
input_3596"�
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
-__inference_model_3595_layer_call_fn_77569139inputs"�
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
-__inference_model_3595_layer_call_fn_77569184inputs"�
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
H__inference_model_3595_layer_call_and_return_conditional_losses_77568602
input_3596"�
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
H__inference_model_3595_layer_call_and_return_conditional_losses_77568656
input_3596"�
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
H__inference_model_3595_layer_call_and_return_conditional_losses_77569253inputs"�
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
H__inference_model_3595_layer_call_and_return_conditional_losses_77569322inputs"�
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
&__inference_signature_wrapper_77569094
input_3596"�
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
.__inference_dense_17970_layer_call_fn_77569331inputs"�
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
I__inference_dense_17970_layer_call_and_return_conditional_losses_77569342inputs"�
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
.__inference_dense_17971_layer_call_fn_77569351inputs"�
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
I__inference_dense_17971_layer_call_and_return_conditional_losses_77569361inputs"�
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
.__inference_dense_17972_layer_call_fn_77569370inputs"�
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
I__inference_dense_17972_layer_call_and_return_conditional_losses_77569381inputs"�
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
.__inference_dense_17973_layer_call_fn_77569390inputs"�
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
I__inference_dense_17973_layer_call_and_return_conditional_losses_77569400inputs"�
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
.__inference_dense_17974_layer_call_fn_77569409inputs"�
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
I__inference_dense_17974_layer_call_and_return_conditional_losses_77569420inputs"�
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
.__inference_dense_17975_layer_call_fn_77569429inputs"�
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
I__inference_dense_17975_layer_call_and_return_conditional_losses_77569439inputs"�
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
.__inference_dense_17976_layer_call_fn_77569448inputs"�
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
I__inference_dense_17976_layer_call_and_return_conditional_losses_77569459inputs"�
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
.__inference_dense_17977_layer_call_fn_77569468inputs"�
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
I__inference_dense_17977_layer_call_and_return_conditional_losses_77569478inputs"�
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
.__inference_dense_17978_layer_call_fn_77569487inputs"�
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
I__inference_dense_17978_layer_call_and_return_conditional_losses_77569498inputs"�
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
.__inference_dense_17979_layer_call_fn_77569507inputs"�
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
I__inference_dense_17979_layer_call_and_return_conditional_losses_77569517inputs"�
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
#__inference__wrapped_model_77568432�#$+,34;<CDKLST[\cd3�0
)�&
$�!

input_3596���������(
� "9�6
4
dense_17979%�"
dense_17979���������(�
I__inference_dense_17970_layer_call_and_return_conditional_losses_77569342c/�,
%�"
 �
inputs���������(
� ",�)
"�
tensor_0���������
� �
.__inference_dense_17970_layer_call_fn_77569331X/�,
%�"
 �
inputs���������(
� "!�
unknown����������
I__inference_dense_17971_layer_call_and_return_conditional_losses_77569361c#$/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������(
� �
.__inference_dense_17971_layer_call_fn_77569351X#$/�,
%�"
 �
inputs���������
� "!�
unknown���������(�
I__inference_dense_17972_layer_call_and_return_conditional_losses_77569381c+,/�,
%�"
 �
inputs���������(
� ",�)
"�
tensor_0���������

� �
.__inference_dense_17972_layer_call_fn_77569370X+,/�,
%�"
 �
inputs���������(
� "!�
unknown���������
�
I__inference_dense_17973_layer_call_and_return_conditional_losses_77569400c34/�,
%�"
 �
inputs���������

� ",�)
"�
tensor_0���������(
� �
.__inference_dense_17973_layer_call_fn_77569390X34/�,
%�"
 �
inputs���������

� "!�
unknown���������(�
I__inference_dense_17974_layer_call_and_return_conditional_losses_77569420c;</�,
%�"
 �
inputs���������(
� ",�)
"�
tensor_0���������
� �
.__inference_dense_17974_layer_call_fn_77569409X;</�,
%�"
 �
inputs���������(
� "!�
unknown����������
I__inference_dense_17975_layer_call_and_return_conditional_losses_77569439cCD/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������(
� �
.__inference_dense_17975_layer_call_fn_77569429XCD/�,
%�"
 �
inputs���������
� "!�
unknown���������(�
I__inference_dense_17976_layer_call_and_return_conditional_losses_77569459cKL/�,
%�"
 �
inputs���������(
� ",�)
"�
tensor_0���������

� �
.__inference_dense_17976_layer_call_fn_77569448XKL/�,
%�"
 �
inputs���������(
� "!�
unknown���������
�
I__inference_dense_17977_layer_call_and_return_conditional_losses_77569478cST/�,
%�"
 �
inputs���������

� ",�)
"�
tensor_0���������(
� �
.__inference_dense_17977_layer_call_fn_77569468XST/�,
%�"
 �
inputs���������

� "!�
unknown���������(�
I__inference_dense_17978_layer_call_and_return_conditional_losses_77569498c[\/�,
%�"
 �
inputs���������(
� ",�)
"�
tensor_0���������
� �
.__inference_dense_17978_layer_call_fn_77569487X[\/�,
%�"
 �
inputs���������(
� "!�
unknown����������
I__inference_dense_17979_layer_call_and_return_conditional_losses_77569517ccd/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������(
� �
.__inference_dense_17979_layer_call_fn_77569507Xcd/�,
%�"
 �
inputs���������
� "!�
unknown���������(�
H__inference_model_3595_layer_call_and_return_conditional_losses_77568602�#$+,34;<CDKLST[\cd;�8
1�.
$�!

input_3596���������(
p

 
� ",�)
"�
tensor_0���������(
� �
H__inference_model_3595_layer_call_and_return_conditional_losses_77568656�#$+,34;<CDKLST[\cd;�8
1�.
$�!

input_3596���������(
p 

 
� ",�)
"�
tensor_0���������(
� �
H__inference_model_3595_layer_call_and_return_conditional_losses_77569253}#$+,34;<CDKLST[\cd7�4
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
H__inference_model_3595_layer_call_and_return_conditional_losses_77569322}#$+,34;<CDKLST[\cd7�4
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
-__inference_model_3595_layer_call_fn_77568756v#$+,34;<CDKLST[\cd;�8
1�.
$�!

input_3596���������(
p

 
� "!�
unknown���������(�
-__inference_model_3595_layer_call_fn_77568855v#$+,34;<CDKLST[\cd;�8
1�.
$�!

input_3596���������(
p 

 
� "!�
unknown���������(�
-__inference_model_3595_layer_call_fn_77569139r#$+,34;<CDKLST[\cd7�4
-�*
 �
inputs���������(
p

 
� "!�
unknown���������(�
-__inference_model_3595_layer_call_fn_77569184r#$+,34;<CDKLST[\cd7�4
-�*
 �
inputs���������(
p 

 
� "!�
unknown���������(�
&__inference_signature_wrapper_77569094�#$+,34;<CDKLST[\cdA�>
� 
7�4
2

input_3596$�!

input_3596���������("9�6
4
dense_17979%�"
dense_17979���������(