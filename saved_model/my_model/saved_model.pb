єн"
й°
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
А
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
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resourceИ
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
В
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
┴
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
executor_typestring Ии
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758вш
~
Adam/outputs/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/outputs/bias/v
w
'Adam/outputs/bias/v/Read/ReadVariableOpReadVariableOpAdam/outputs/bias/v*
_output_shapes
:*
dtype0
З
Adam/outputs/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*&
shared_nameAdam/outputs/kernel/v
А
)Adam/outputs/kernel/v/Read/ReadVariableOpReadVariableOpAdam/outputs/kernel/v*
_output_shapes
:	А*
dtype0
{
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameAdam/dense/bias/v
t
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes	
:А*
dtype0
Е
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АЄА*$
shared_nameAdam/dense/kernel/v
~
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*!
_output_shapes
:АЄА*
dtype0
А
Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_2/bias/v
y
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes
:@*
dtype0
Р
Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2d_2/kernel/v
Й
*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*&
_output_shapes
: @*
dtype0
А
Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
: *
dtype0
Р
Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_1/kernel/v
Й
*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
: *
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
:*
dtype0
М
Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/v
Е
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:*
dtype0
~
Adam/outputs/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/outputs/bias/m
w
'Adam/outputs/bias/m/Read/ReadVariableOpReadVariableOpAdam/outputs/bias/m*
_output_shapes
:*
dtype0
З
Adam/outputs/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*&
shared_nameAdam/outputs/kernel/m
А
)Adam/outputs/kernel/m/Read/ReadVariableOpReadVariableOpAdam/outputs/kernel/m*
_output_shapes
:	А*
dtype0
{
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameAdam/dense/bias/m
t
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes	
:А*
dtype0
Е
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АЄА*$
shared_nameAdam/dense/kernel/m
~
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*!
_output_shapes
:АЄА*
dtype0
А
Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_2/bias/m
y
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
_output_shapes
:@*
dtype0
Р
Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2d_2/kernel/m
Й
*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*&
_output_shapes
: @*
dtype0
А
Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
: *
dtype0
Р
Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_1/kernel/m
Й
*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
: *
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
:*
dtype0
М
Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/m
Е
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:*
dtype0
А
random_zoom/StateVarVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*%
shared_namerandom_zoom/StateVar
y
(random_zoom/StateVar/Read/ReadVariableOpReadVariableOprandom_zoom/StateVar*
_output_shapes
:*
dtype0	
И
random_rotation/StateVarVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*)
shared_namerandom_rotation/StateVar
Б
,random_rotation/StateVar/Read/ReadVariableOpReadVariableOprandom_rotation/StateVar*
_output_shapes
:*
dtype0	
А
random_flip/StateVarVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*%
shared_namerandom_flip/StateVar
y
(random_flip/StateVar/Read/ReadVariableOpReadVariableOprandom_flip/StateVar*
_output_shapes
:*
dtype0	
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
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
p
outputs/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutputs/bias
i
 outputs/bias/Read/ReadVariableOpReadVariableOpoutputs/bias*
_output_shapes
:*
dtype0
y
outputs/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*
shared_nameoutputs/kernel
r
"outputs/kernel/Read/ReadVariableOpReadVariableOpoutputs/kernel*
_output_shapes
:	А*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:А*
dtype0
w
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АЄА*
shared_namedense/kernel
p
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*!
_output_shapes
:АЄА*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:@*
dtype0
В
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
: *
dtype0
В
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
: *
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
Ы
$serving_default_module_wrapper_inputPlaceholder*1
_output_shapes
:         ┤┤*
dtype0*&
shape:         ┤┤
х
StatefulPartitionedCallStatefulPartitionedCall$serving_default_module_wrapper_inputconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasdense/kernel
dense/biasoutputs/kerneloutputs/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference_signature_wrapper_8046

NoOpNoOp
┴}
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*№|
valueЄ|Bя| Bш|
ь
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
layer_with_weights-4
layer-11
	variables
trainable_variables
regularization_losses
	keras_api
*&call_and_return_all_conditional_losses
__call__
_default_save_signature
	optimizer

signatures*
Э
	variables
regularization_losses
trainable_variables
	keras_api
__call__
*&call_and_return_all_conditional_losses
_module*
Ы
	variables
regularization_losses
trainable_variables
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
#_module* 
ж
$	variables
%regularization_losses
&trainable_variables
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

*kernel
+bias*
О
,	variables
-regularization_losses
.trainable_variables
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses* 
ж
2	variables
3regularization_losses
4trainable_variables
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses

8kernel
9bias*
О
:	variables
;regularization_losses
<trainable_variables
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses* 
ж
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

Fkernel
Gbias*
О
H	variables
Iregularization_losses
Jtrainable_variables
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses* 
О
N	variables
Oregularization_losses
Ptrainable_variables
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses* 
О
T	variables
Uregularization_losses
Vtrainable_variables
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses* 
ж
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses

`kernel
abias*
ж
b	variables
cregularization_losses
dtrainable_variables
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

hkernel
ibias*
J
*0
+1
82
93
F4
G5
`6
a7
h8
i9*
J
*0
+1
82
93
F4
G5
`6
a7
h8
i9*
* 
░
jlayer_metrics
	variables

klayers
lnon_trainable_variables
mlayer_regularization_losses
nmetrics
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
otrace_0
ptrace_1
qtrace_2
rtrace_3* 
6
strace_0
ttrace_1
utrace_2
vtrace_3* 

wtrace_0* 
М
xiter

ybeta_1

zbeta_2
	{decay
|learning_rate*m║+m╗8m╝9m╜Fm╛Gm┐`m└am┴hm┬im├*v─+v┼8v╞9v╟Fv╚Gv╔`v╩av╦hv╠iv═*

}serving_default* 
* 
* 
* 
Ф
~layer_metrics
	variables
non_trainable_variables
regularization_losses
 Аlayer_regularization_losses
Бmetrics
trainable_variables
Вlayers
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

Гtrace_0
Дtrace_1* 

Еtrace_0
Жtrace_1* 
└
Зlayer-0
Иlayer-1
Йlayer-2
К	variables
Лtrainable_variables
Мregularization_losses
Н	keras_api
О__call__
+П&call_and_return_all_conditional_losses*
* 
* 
* 
Ц
Рlayer_metrics
	variables
Сnon_trainable_variables
regularization_losses
 Тlayer_regularization_losses
Уmetrics
trainable_variables
Фlayers
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses* 

Хtrace_0
Цtrace_1* 

Чtrace_0
Шtrace_1* 
Ф
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Ь	keras_api
Э__call__
+Ю&call_and_return_all_conditional_losses* 

*0
+1*
* 

*0
+1*
Ш
Яlayer_metrics
$	variables
аnon_trainable_variables
%regularization_losses
 бlayer_regularization_losses
вmetrics
&trainable_variables
гlayers
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*

дtrace_0* 

еtrace_0* 
]W
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
жlayer_metrics
,	variables
зnon_trainable_variables
-regularization_losses
 иlayer_regularization_losses
йmetrics
.trainable_variables
кlayers
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses* 

лtrace_0* 

мtrace_0* 

80
91*
* 

80
91*
Ш
нlayer_metrics
2	variables
оnon_trainable_variables
3regularization_losses
 пlayer_regularization_losses
░metrics
4trainable_variables
▒layers
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses*

▓trace_0* 

│trace_0* 
_Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
┤layer_metrics
:	variables
╡non_trainable_variables
;regularization_losses
 ╢layer_regularization_losses
╖metrics
<trainable_variables
╕layers
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses* 

╣trace_0* 

║trace_0* 

F0
G1*
* 

F0
G1*
Ш
╗layer_metrics
@	variables
╝non_trainable_variables
Aregularization_losses
 ╜layer_regularization_losses
╛metrics
Btrainable_variables
┐layers
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses*

└trace_0* 

┴trace_0* 
_Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
┬layer_metrics
H	variables
├non_trainable_variables
Iregularization_losses
 ─layer_regularization_losses
┼metrics
Jtrainable_variables
╞layers
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses* 

╟trace_0* 

╚trace_0* 
* 
* 
* 
Ц
╔layer_metrics
N	variables
╩non_trainable_variables
Oregularization_losses
 ╦layer_regularization_losses
╠metrics
Ptrainable_variables
═layers
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses* 

╬trace_0
╧trace_1* 

╨trace_0
╤trace_1* 
* 
* 
* 
Ц
╥layer_metrics
T	variables
╙non_trainable_variables
Uregularization_losses
 ╘layer_regularization_losses
╒metrics
Vtrainable_variables
╓layers
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses* 

╫trace_0* 

╪trace_0* 

`0
a1*
* 

`0
a1*
Ш
┘layer_metrics
Z	variables
┌non_trainable_variables
[regularization_losses
 █layer_regularization_losses
▄metrics
\trainable_variables
▌layers
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses*

▐trace_0* 

▀trace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

h0
i1*
* 

h0
i1*
Ш
рlayer_metrics
b	variables
сnon_trainable_variables
cregularization_losses
 тlayer_regularization_losses
уmetrics
dtrainable_variables
фlayers
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*

хtrace_0* 

цtrace_0* 
^X
VARIABLE_VALUEoutputs/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEoutputs/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
Z
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
10
11*
* 
* 

ч0
ш1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
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
о
щ	variables
ъtrainable_variables
ыregularization_losses
ь	keras_api
э__call__
+ю&call_and_return_all_conditional_losses
я_random_generator*
о
Ё	variables
ёtrainable_variables
Єregularization_losses
є	keras_api
Ї__call__
+ї&call_and_return_all_conditional_losses
Ў_random_generator*
о
ў	variables
°trainable_variables
∙regularization_losses
·	keras_api
√__call__
+№&call_and_return_all_conditional_losses
¤_random_generator*
* 
* 
* 
Ю
■non_trainable_variables
 layers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
К	variables
Лtrainable_variables
Мregularization_losses
О__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses*
:
Гtrace_0
Дtrace_1
Еtrace_2
Жtrace_3* 
:
Зtrace_0
Иtrace_1
Йtrace_2
Кtrace_3* 
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
Ь
Лnon_trainable_variables
Мlayers
Нmetrics
 Оlayer_regularization_losses
Пlayer_metrics
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Э__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses* 
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
* 
* 
<
Р	variables
С	keras_api

Тtotal

Уcount*
M
Ф	variables
Х	keras_api

Цtotal

Чcount
Ш
_fn_kwargs*
* 
* 
* 
Ь
Щnon_trainable_variables
Ъlayers
Ыmetrics
 Ьlayer_regularization_losses
Эlayer_metrics
щ	variables
ъtrainable_variables
ыregularization_losses
э__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses* 

Юtrace_0
Яtrace_1* 

аtrace_0
бtrace_1* 

в
_generator*
* 
* 
* 
Ь
гnon_trainable_variables
дlayers
еmetrics
 жlayer_regularization_losses
зlayer_metrics
Ё	variables
ёtrainable_variables
Єregularization_losses
Ї__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses* 

иtrace_0
йtrace_1* 

кtrace_0
лtrace_1* 

м
_generator*
* 
* 
* 
Ь
нnon_trainable_variables
оlayers
пmetrics
 ░layer_regularization_losses
▒layer_metrics
ў	variables
°trainable_variables
∙regularization_losses
√__call__
+№&call_and_return_all_conditional_losses
'№"call_and_return_conditional_losses* 

▓trace_0
│trace_1* 

┤trace_0
╡trace_1* 

╢
_generator*
* 

З0
И1
Й2*
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

Т0
У1*

Р	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Ц0
Ч1*

Ф	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
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

╖
_state_var*
* 
* 
* 
* 
* 
* 
* 
* 
* 

╕
_state_var*
* 
* 
* 
* 
* 
* 
* 
* 
* 

╣
_state_var*
ЙВ
VARIABLE_VALUErandom_flip/StateVarZlayer-0/_module/layer-0/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUErandom_rotation/StateVarZlayer-0/_module/layer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUErandom_zoom/StateVarZlayer-0/_module/layer-2/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv2d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv2d_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdam/outputs/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/outputs/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv2d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv2d_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdam/outputs/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/outputs/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
╖
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasdense/kernel
dense/biasoutputs/kerneloutputs/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountrandom_flip/StateVarrandom_rotation/StateVarrandom_zoom/StateVarAdam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/outputs/kernel/mAdam/outputs/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/vAdam/dense/kernel/vAdam/dense/bias/vAdam/outputs/kernel/vAdam/outputs/bias/vConst*7
Tin0
.2,*
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
GPU 2J 8В *'
f"R 
__inference__traced_save_10236
▓
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasdense/kernel
dense/biasoutputs/kerneloutputs/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountrandom_flip/StateVarrandom_rotation/StateVarrandom_zoom/StateVarAdam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/outputs/kernel/mAdam/outputs/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/vAdam/dense/kernel/vAdam/dense/bias/vAdam/outputs/kernel/vAdam/outputs/bias/v*6
Tin/
-2+*
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
GPU 2J 8В **
f%R#
!__inference__traced_restore_10372зД
▒/
с
D__inference_sequential_layer_call_and_return_conditional_losses_7971
module_wrapper_input%
conv2d_7940:
conv2d_7942:'
conv2d_1_7946: 
conv2d_1_7948: '
conv2d_2_7952: @
conv2d_2_7954:@

dense_7960:АЄА

dense_7962:	А
outputs_7965:	А
outputs_7967:
identityИвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallв conv2d_2/StatefulPartitionedCallвdense/StatefulPartitionedCallвoutputs/StatefulPartitionedCall┌
module_wrapper/PartitionedCallPartitionedCallmodule_wrapper_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_module_wrapper_layer_call_and_return_conditional_losses_7283ё
 module_wrapper_1/PartitionedCallPartitionedCall'module_wrapper/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_7293П
conv2d/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_1/PartitionedCall:output:0conv2d_7940conv2d_7942*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_7306щ
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ZZ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_7242Т
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_7946conv2d_1_7948*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ZZ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_7324я
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         -- * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_7254Ф
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_7952conv2d_2_7954*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         --@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_7342я
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_7266▐
dropout/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_7354╨
flatten/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         АЄ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_7362∙
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dense_7960
dense_7962*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_7375Ж
outputs/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0outputs_7965outputs_7967*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_outputs_layer_call_and_return_conditional_losses_7391w
IdentityIdentity(outputs/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         я
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^outputs/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:         ┤┤: : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
outputs/StatefulPartitionedCalloutputs/StatefulPartitionedCall:g c
1
_output_shapes
:         ┤┤
.
_user_specified_namemodule_wrapper_input
─
~
.__inference_random_rotation_layer_call_fn_9701

inputs
unknown:	
identityИвStatefulPartitionedCall╪
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_random_rotation_layer_call_and_return_conditional_losses_8945y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ┤┤`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ┤┤: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameinputs
▓n
╛
E__inference_random_zoom_layer_call_and_return_conditional_losses_9942

inputs6
(stateful_uniform_rngreadandskip_resource:	
identityИвstateful_uniform/RngReadAndSkipI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
¤        j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
■        a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: Z
stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :З
stateful_uniform/shapePackstrided_slice:output:0!stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:Y
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *fff?Y
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *═╠М?`
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: А
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: Y
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :o
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ╢
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:n
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:м
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask}
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0p
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:в
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:Б
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0o
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :П
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:         z
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: Ы
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*'
_output_shapes
:         Д
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*'
_output_shapes
:         M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :П
concatConcatV2stateful_uniform:z:0stateful_uniform:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:         ^
zoom_matrix/ShapeShapeconcat:output:0*
T0*
_output_shapes
::э╧i
zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
zoom_matrix/strided_sliceStridedSlicezoom_matrix/Shape:output:0(zoom_matrix/strided_slice/stack:output:0*zoom_matrix/strided_slice/stack_1:output:0*zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?_
zoom_matrix/subSub
Cast_1:y:0zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: Z
zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @t
zoom_matrix/truedivRealDivzoom_matrix/sub:z:0zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: v
!zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            x
#zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╥
zoom_matrix/strided_slice_1StridedSliceconcat:output:0*zoom_matrix/strided_slice_1/stack:output:0,zoom_matrix/strided_slice_1/stack_1:output:0,zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskX
zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?О
zoom_matrix/sub_1Subzoom_matrix/sub_1/x:output:0$zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:         x
zoom_matrix/mulMulzoom_matrix/truediv:z:0zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:         X
zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?a
zoom_matrix/sub_2SubCast:y:0zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: \
zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @z
zoom_matrix/truediv_1RealDivzoom_matrix/sub_2:z:0 zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: v
!zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╥
zoom_matrix/strided_slice_2StridedSliceconcat:output:0*zoom_matrix/strided_slice_2/stack:output:0,zoom_matrix/strided_slice_2/stack_1:output:0,zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskX
zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?О
zoom_matrix/sub_3Subzoom_matrix/sub_3/x:output:0$zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:         |
zoom_matrix/mul_1Mulzoom_matrix/truediv_1:z:0zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:         v
!zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            x
#zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╥
zoom_matrix/strided_slice_3StridedSliceconcat:output:0*zoom_matrix/strided_slice_3/stack:output:0,zoom_matrix/strided_slice_3/stack_1:output:0,zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask\
zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ч
zoom_matrix/zeros/packedPack"zoom_matrix/strided_slice:output:0#zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:\
zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Р
zoom_matrix/zerosFill!zoom_matrix/zeros/packed:output:0 zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:         ^
zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ы
zoom_matrix/zeros_1/packedPack"zoom_matrix/strided_slice:output:0%zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:^
zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ц
zoom_matrix/zeros_1Fill#zoom_matrix/zeros_1/packed:output:0"zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:         v
!zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╥
zoom_matrix/strided_slice_4StridedSliceconcat:output:0*zoom_matrix/strided_slice_4/stack:output:0,zoom_matrix/strided_slice_4/stack_1:output:0,zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask^
zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ы
zoom_matrix/zeros_2/packedPack"zoom_matrix/strided_slice:output:0%zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:^
zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ц
zoom_matrix/zeros_2Fill#zoom_matrix/zeros_2/packed:output:0"zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:         Y
zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :╦
zoom_matrix/concatConcatV2$zoom_matrix/strided_slice_3:output:0zoom_matrix/zeros:output:0zoom_matrix/mul:z:0zoom_matrix/zeros_1:output:0$zoom_matrix/strided_slice_4:output:0zoom_matrix/mul_1:z:0zoom_matrix/zeros_2:output:0 zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:         S
transform/ShapeShapeinputs*
T0*
_output_shapes
::э╧g
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:я
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:Y
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    Э
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputszoom_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*1
_output_shapes
:         ┤┤*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEARТ
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:         ┤┤h
NoOpNoOp ^stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ┤┤: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameinputs
╟
F
*__inference_random_flip_layer_call_fn_9632

inputs
identity║
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_random_flip_layer_call_and_return_conditional_losses_9062j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ┤┤"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ┤┤:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameinputs
ж

Ї
?__inference_dense_layer_call_and_return_conditional_losses_9311

inputs3
matmul_readvariableop_resource:АЄА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpw
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:АЄА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         АЄ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:         АЄ
 
_user_specified_nameinputs
л
B
&__inference_flatten_layer_call_fn_9285

inputs
identityо
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         АЄ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_7362b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:         АЄ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
∙з
Щ
H__inference_module_wrapper_layer_call_and_return_conditional_losses_7797

args_0V
Hsequential_random_flip_stateful_uniform_full_int_rngreadandskip_resource:	Q
Csequential_random_rotation_stateful_uniform_rngreadandskip_resource:	M
?sequential_random_zoom_stateful_uniform_rngreadandskip_resource:	
identityИв?sequential/random_flip/stateful_uniform_full_int/RngReadAndSkipв:sequential/random_rotation/stateful_uniform/RngReadAndSkipв6sequential/random_zoom/stateful_uniform/RngReadAndSkipА
6sequential/random_flip/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:А
6sequential/random_flip/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: р
5sequential/random_flip/stateful_uniform_full_int/ProdProd?sequential/random_flip/stateful_uniform_full_int/shape:output:0?sequential/random_flip/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: y
7sequential/random_flip/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :п
7sequential/random_flip/stateful_uniform_full_int/Cast_1Cast>sequential/random_flip/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ╢
?sequential/random_flip/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkipHsequential_random_flip_stateful_uniform_full_int_rngreadandskip_resource@sequential/random_flip/stateful_uniform_full_int/Cast/x:output:0;sequential/random_flip/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:О
Dsequential/random_flip/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Р
Fsequential/random_flip/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Р
Fsequential/random_flip/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╠
>sequential/random_flip/stateful_uniform_full_int/strided_sliceStridedSliceGsequential/random_flip/stateful_uniform_full_int/RngReadAndSkip:value:0Msequential/random_flip/stateful_uniform_full_int/strided_slice/stack:output:0Osequential/random_flip/stateful_uniform_full_int/strided_slice/stack_1:output:0Osequential/random_flip/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask╜
8sequential/random_flip/stateful_uniform_full_int/BitcastBitcastGsequential/random_flip/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0Р
Fsequential/random_flip/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Т
Hsequential/random_flip/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Т
Hsequential/random_flip/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┬
@sequential/random_flip/stateful_uniform_full_int/strided_slice_1StridedSliceGsequential/random_flip/stateful_uniform_full_int/RngReadAndSkip:value:0Osequential/random_flip/stateful_uniform_full_int/strided_slice_1/stack:output:0Qsequential/random_flip/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Qsequential/random_flip/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:┴
:sequential/random_flip/stateful_uniform_full_int/Bitcast_1BitcastIsequential/random_flip/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0v
4sequential/random_flip/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :Д
0sequential/random_flip/stateful_uniform_full_intStatelessRandomUniformFullIntV2?sequential/random_flip/stateful_uniform_full_int/shape:output:0Csequential/random_flip/stateful_uniform_full_int/Bitcast_1:output:0Asequential/random_flip/stateful_uniform_full_int/Bitcast:output:0=sequential/random_flip/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	k
!sequential/random_flip/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R ╜
sequential/random_flip/stackPack9sequential/random_flip/stateful_uniform_full_int:output:0*sequential/random_flip/zeros_like:output:0*
N*
T0	*
_output_shapes

:{
*sequential/random_flip/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        }
,sequential/random_flip/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,sequential/random_flip/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ъ
$sequential/random_flip/strided_sliceStridedSlice%sequential/random_flip/stack:output:03sequential/random_flip/strided_slice/stack:output:05sequential/random_flip/strided_slice/stack_1:output:05sequential/random_flip/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask╡
Jsequential/random_flip/stateless_random_flip_left_right/control_dependencyIdentityargs_0*
T0*
_class
loc:@args_0*1
_output_shapes
:         ┤┤╬
=sequential/random_flip/stateless_random_flip_left_right/ShapeShapeSsequential/random_flip/stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
::э╧Х
Ksequential/random_flip/stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ч
Msequential/random_flip/stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ч
Msequential/random_flip/stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
Esequential/random_flip/stateless_random_flip_left_right/strided_sliceStridedSliceFsequential/random_flip/stateless_random_flip_left_right/Shape:output:0Tsequential/random_flip/stateless_random_flip_left_right/strided_slice/stack:output:0Vsequential/random_flip/stateless_random_flip_left_right/strided_slice/stack_1:output:0Vsequential/random_flip/stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask▄
Vsequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/shapePackNsequential/random_flip/stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:Щ
Tsequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    Щ
Tsequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  А?▐
msequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter-sequential/random_flip/strided_slice:output:0* 
_output_shapes
::п
msequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :ё
isequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2_sequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/shape:output:0ssequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0wsequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0vsequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:         ║
Tsequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/subSub]sequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/max:output:0]sequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ╫
Tsequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/mulMulrsequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Xsequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:         └
Psequential/random_flip/stateless_random_flip_left_right/stateless_random_uniformAddV2Xsequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/mul:z:0]sequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:         Й
Gsequential/random_flip/stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Й
Gsequential/random_flip/stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Й
Gsequential/random_flip/stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :┴
Esequential/random_flip/stateless_random_flip_left_right/Reshape/shapePackNsequential/random_flip/stateless_random_flip_left_right/strided_slice:output:0Psequential/random_flip/stateless_random_flip_left_right/Reshape/shape/1:output:0Psequential/random_flip/stateless_random_flip_left_right/Reshape/shape/2:output:0Psequential/random_flip/stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:к
?sequential/random_flip/stateless_random_flip_left_right/ReshapeReshapeTsequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform:z:0Nsequential/random_flip/stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:         ╩
=sequential/random_flip/stateless_random_flip_left_right/RoundRoundHsequential/random_flip/stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:         Р
Fsequential/random_flip/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:░
Asequential/random_flip/stateless_random_flip_left_right/ReverseV2	ReverseV2Ssequential/random_flip/stateless_random_flip_left_right/control_dependency:output:0Osequential/random_flip/stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:         ┤┤Н
;sequential/random_flip/stateless_random_flip_left_right/mulMulAsequential/random_flip/stateless_random_flip_left_right/Round:y:0Jsequential/random_flip/stateless_random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:         ┤┤В
=sequential/random_flip/stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?З
;sequential/random_flip/stateless_random_flip_left_right/subSubFsequential/random_flip/stateless_random_flip_left_right/sub/x:output:0Asequential/random_flip/stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:         Ц
=sequential/random_flip/stateless_random_flip_left_right/mul_1Mul?sequential/random_flip/stateless_random_flip_left_right/sub:z:0Ssequential/random_flip/stateless_random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:         ┤┤Д
;sequential/random_flip/stateless_random_flip_left_right/addAddV2?sequential/random_flip/stateless_random_flip_left_right/mul:z:0Asequential/random_flip/stateless_random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:         ┤┤Э
 sequential/random_rotation/ShapeShape?sequential/random_flip/stateless_random_flip_left_right/add:z:0*
T0*
_output_shapes
::э╧x
.sequential/random_rotation/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0sequential/random_rotation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0sequential/random_rotation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╪
(sequential/random_rotation/strided_sliceStridedSlice)sequential/random_rotation/Shape:output:07sequential/random_rotation/strided_slice/stack:output:09sequential/random_rotation/strided_slice/stack_1:output:09sequential/random_rotation/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskГ
0sequential/random_rotation/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
¤        Е
2sequential/random_rotation/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
■        |
2sequential/random_rotation/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:р
*sequential/random_rotation/strided_slice_1StridedSlice)sequential/random_rotation/Shape:output:09sequential/random_rotation/strided_slice_1/stack:output:0;sequential/random_rotation/strided_slice_1/stack_1:output:0;sequential/random_rotation/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskМ
sequential/random_rotation/CastCast3sequential/random_rotation/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: Г
0sequential/random_rotation/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        Е
2sequential/random_rotation/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         |
2sequential/random_rotation/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:р
*sequential/random_rotation/strided_slice_2StridedSlice)sequential/random_rotation/Shape:output:09sequential/random_rotation/strided_slice_2/stack:output:0;sequential/random_rotation/strided_slice_2/stack_1:output:0;sequential/random_rotation/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskО
!sequential/random_rotation/Cast_1Cast3sequential/random_rotation/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: Ъ
1sequential/random_rotation/stateful_uniform/shapePack1sequential/random_rotation/strided_slice:output:0*
N*
T0*
_output_shapes
:t
/sequential/random_rotation/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|┘ ┐t
/sequential/random_rotation/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|┘ ?{
1sequential/random_rotation/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ╤
0sequential/random_rotation/stateful_uniform/ProdProd:sequential/random_rotation/stateful_uniform/shape:output:0:sequential/random_rotation/stateful_uniform/Const:output:0*
T0*
_output_shapes
: t
2sequential/random_rotation/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :е
2sequential/random_rotation/stateful_uniform/Cast_1Cast9sequential/random_rotation/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: в
:sequential/random_rotation/stateful_uniform/RngReadAndSkipRngReadAndSkipCsequential_random_rotation_stateful_uniform_rngreadandskip_resource;sequential/random_rotation/stateful_uniform/Cast/x:output:06sequential/random_rotation/stateful_uniform/Cast_1:y:0*
_output_shapes
:Й
?sequential/random_rotation/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Л
Asequential/random_rotation/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Л
Asequential/random_rotation/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:│
9sequential/random_rotation/stateful_uniform/strided_sliceStridedSliceBsequential/random_rotation/stateful_uniform/RngReadAndSkip:value:0Hsequential/random_rotation/stateful_uniform/strided_slice/stack:output:0Jsequential/random_rotation/stateful_uniform/strided_slice/stack_1:output:0Jsequential/random_rotation/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask│
3sequential/random_rotation/stateful_uniform/BitcastBitcastBsequential/random_rotation/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0Л
Asequential/random_rotation/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Н
Csequential/random_rotation/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Н
Csequential/random_rotation/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
;sequential/random_rotation/stateful_uniform/strided_slice_1StridedSliceBsequential/random_rotation/stateful_uniform/RngReadAndSkip:value:0Jsequential/random_rotation/stateful_uniform/strided_slice_1/stack:output:0Lsequential/random_rotation/stateful_uniform/strided_slice_1/stack_1:output:0Lsequential/random_rotation/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:╖
5sequential/random_rotation/stateful_uniform/Bitcast_1BitcastDsequential/random_rotation/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0К
Hsequential/random_rotation/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :Т
Dsequential/random_rotation/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2:sequential/random_rotation/stateful_uniform/shape:output:0>sequential/random_rotation/stateful_uniform/Bitcast_1:output:0<sequential/random_rotation/stateful_uniform/Bitcast:output:0Qsequential/random_rotation/stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:         ╦
/sequential/random_rotation/stateful_uniform/subSub8sequential/random_rotation/stateful_uniform/max:output:08sequential/random_rotation/stateful_uniform/min:output:0*
T0*
_output_shapes
: ш
/sequential/random_rotation/stateful_uniform/mulMulMsequential/random_rotation/stateful_uniform/StatelessRandomUniformV2:output:03sequential/random_rotation/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:         ╤
+sequential/random_rotation/stateful_uniformAddV23sequential/random_rotation/stateful_uniform/mul:z:08sequential/random_rotation/stateful_uniform/min:output:0*
T0*#
_output_shapes
:         u
0sequential/random_rotation/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╕
.sequential/random_rotation/rotation_matrix/subSub%sequential/random_rotation/Cast_1:y:09sequential/random_rotation/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: Ф
.sequential/random_rotation/rotation_matrix/CosCos/sequential/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:         w
2sequential/random_rotation/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╝
0sequential/random_rotation/rotation_matrix/sub_1Sub%sequential/random_rotation/Cast_1:y:0;sequential/random_rotation/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: ═
.sequential/random_rotation/rotation_matrix/mulMul2sequential/random_rotation/rotation_matrix/Cos:y:04sequential/random_rotation/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:         Ф
.sequential/random_rotation/rotation_matrix/SinSin/sequential/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:         w
2sequential/random_rotation/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?║
0sequential/random_rotation/rotation_matrix/sub_2Sub#sequential/random_rotation/Cast:y:0;sequential/random_rotation/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: ╧
0sequential/random_rotation/rotation_matrix/mul_1Mul2sequential/random_rotation/rotation_matrix/Sin:y:04sequential/random_rotation/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:         ╧
0sequential/random_rotation/rotation_matrix/sub_3Sub2sequential/random_rotation/rotation_matrix/mul:z:04sequential/random_rotation/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:         ╧
0sequential/random_rotation/rotation_matrix/sub_4Sub2sequential/random_rotation/rotation_matrix/sub:z:04sequential/random_rotation/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:         y
4sequential/random_rotation/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @р
2sequential/random_rotation/rotation_matrix/truedivRealDiv4sequential/random_rotation/rotation_matrix/sub_4:z:0=sequential/random_rotation/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:         w
2sequential/random_rotation/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?║
0sequential/random_rotation/rotation_matrix/sub_5Sub#sequential/random_rotation/Cast:y:0;sequential/random_rotation/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: Ц
0sequential/random_rotation/rotation_matrix/Sin_1Sin/sequential/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:         w
2sequential/random_rotation/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╝
0sequential/random_rotation/rotation_matrix/sub_6Sub%sequential/random_rotation/Cast_1:y:0;sequential/random_rotation/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: ╤
0sequential/random_rotation/rotation_matrix/mul_2Mul4sequential/random_rotation/rotation_matrix/Sin_1:y:04sequential/random_rotation/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:         Ц
0sequential/random_rotation/rotation_matrix/Cos_1Cos/sequential/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:         w
2sequential/random_rotation/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?║
0sequential/random_rotation/rotation_matrix/sub_7Sub#sequential/random_rotation/Cast:y:0;sequential/random_rotation/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: ╤
0sequential/random_rotation/rotation_matrix/mul_3Mul4sequential/random_rotation/rotation_matrix/Cos_1:y:04sequential/random_rotation/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:         ╤
.sequential/random_rotation/rotation_matrix/addAddV24sequential/random_rotation/rotation_matrix/mul_2:z:04sequential/random_rotation/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:         ╧
0sequential/random_rotation/rotation_matrix/sub_8Sub4sequential/random_rotation/rotation_matrix/sub_5:z:02sequential/random_rotation/rotation_matrix/add:z:0*
T0*#
_output_shapes
:         {
6sequential/random_rotation/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @ф
4sequential/random_rotation/rotation_matrix/truediv_1RealDiv4sequential/random_rotation/rotation_matrix/sub_8:z:0?sequential/random_rotation/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:         Э
0sequential/random_rotation/rotation_matrix/ShapeShape/sequential/random_rotation/stateful_uniform:z:0*
T0*
_output_shapes
::э╧И
>sequential/random_rotation/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: К
@sequential/random_rotation/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:К
@sequential/random_rotation/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:и
8sequential/random_rotation/rotation_matrix/strided_sliceStridedSlice9sequential/random_rotation/rotation_matrix/Shape:output:0Gsequential/random_rotation/rotation_matrix/strided_slice/stack:output:0Isequential/random_rotation/rotation_matrix/strided_slice/stack_1:output:0Isequential/random_rotation/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЦ
0sequential/random_rotation/rotation_matrix/Cos_2Cos/sequential/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:         С
@sequential/random_rotation/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        У
Bsequential/random_rotation/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        У
Bsequential/random_rotation/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      █
:sequential/random_rotation/rotation_matrix/strided_slice_1StridedSlice4sequential/random_rotation/rotation_matrix/Cos_2:y:0Isequential/random_rotation/rotation_matrix/strided_slice_1/stack:output:0Ksequential/random_rotation/rotation_matrix/strided_slice_1/stack_1:output:0Ksequential/random_rotation/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_maskЦ
0sequential/random_rotation/rotation_matrix/Sin_2Sin/sequential/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:         С
@sequential/random_rotation/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        У
Bsequential/random_rotation/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        У
Bsequential/random_rotation/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      █
:sequential/random_rotation/rotation_matrix/strided_slice_2StridedSlice4sequential/random_rotation/rotation_matrix/Sin_2:y:0Isequential/random_rotation/rotation_matrix/strided_slice_2/stack:output:0Ksequential/random_rotation/rotation_matrix/strided_slice_2/stack_1:output:0Ksequential/random_rotation/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_maskм
.sequential/random_rotation/rotation_matrix/NegNegCsequential/random_rotation/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:         С
@sequential/random_rotation/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        У
Bsequential/random_rotation/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        У
Bsequential/random_rotation/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ▌
:sequential/random_rotation/rotation_matrix/strided_slice_3StridedSlice6sequential/random_rotation/rotation_matrix/truediv:z:0Isequential/random_rotation/rotation_matrix/strided_slice_3/stack:output:0Ksequential/random_rotation/rotation_matrix/strided_slice_3/stack_1:output:0Ksequential/random_rotation/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_maskЦ
0sequential/random_rotation/rotation_matrix/Sin_3Sin/sequential/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:         С
@sequential/random_rotation/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        У
Bsequential/random_rotation/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        У
Bsequential/random_rotation/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      █
:sequential/random_rotation/rotation_matrix/strided_slice_4StridedSlice4sequential/random_rotation/rotation_matrix/Sin_3:y:0Isequential/random_rotation/rotation_matrix/strided_slice_4/stack:output:0Ksequential/random_rotation/rotation_matrix/strided_slice_4/stack_1:output:0Ksequential/random_rotation/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_maskЦ
0sequential/random_rotation/rotation_matrix/Cos_3Cos/sequential/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:         С
@sequential/random_rotation/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        У
Bsequential/random_rotation/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        У
Bsequential/random_rotation/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      █
:sequential/random_rotation/rotation_matrix/strided_slice_5StridedSlice4sequential/random_rotation/rotation_matrix/Cos_3:y:0Isequential/random_rotation/rotation_matrix/strided_slice_5/stack:output:0Ksequential/random_rotation/rotation_matrix/strided_slice_5/stack_1:output:0Ksequential/random_rotation/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_maskС
@sequential/random_rotation/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        У
Bsequential/random_rotation/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        У
Bsequential/random_rotation/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ▀
:sequential/random_rotation/rotation_matrix/strided_slice_6StridedSlice8sequential/random_rotation/rotation_matrix/truediv_1:z:0Isequential/random_rotation/rotation_matrix/strided_slice_6/stack:output:0Ksequential/random_rotation/rotation_matrix/strided_slice_6/stack_1:output:0Ksequential/random_rotation/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask{
9sequential/random_rotation/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ї
7sequential/random_rotation/rotation_matrix/zeros/packedPackAsequential/random_rotation/rotation_matrix/strided_slice:output:0Bsequential/random_rotation/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:{
6sequential/random_rotation/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    э
0sequential/random_rotation/rotation_matrix/zerosFill@sequential/random_rotation/rotation_matrix/zeros/packed:output:0?sequential/random_rotation/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:         x
6sequential/random_rotation/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Б
1sequential/random_rotation/rotation_matrix/concatConcatV2Csequential/random_rotation/rotation_matrix/strided_slice_1:output:02sequential/random_rotation/rotation_matrix/Neg:y:0Csequential/random_rotation/rotation_matrix/strided_slice_3:output:0Csequential/random_rotation/rotation_matrix/strided_slice_4:output:0Csequential/random_rotation/rotation_matrix/strided_slice_5:output:0Csequential/random_rotation/rotation_matrix/strided_slice_6:output:09sequential/random_rotation/rotation_matrix/zeros:output:0?sequential/random_rotation/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:         з
*sequential/random_rotation/transform/ShapeShape?sequential/random_flip/stateless_random_flip_left_right/add:z:0*
T0*
_output_shapes
::э╧В
8sequential/random_rotation/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:Д
:sequential/random_rotation/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Д
:sequential/random_rotation/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ў
2sequential/random_rotation/transform/strided_sliceStridedSlice3sequential/random_rotation/transform/Shape:output:0Asequential/random_rotation/transform/strided_slice/stack:output:0Csequential/random_rotation/transform/strided_slice/stack_1:output:0Csequential/random_rotation/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:t
/sequential/random_rotation/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ╞
?sequential/random_rotation/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3?sequential/random_flip/stateless_random_flip_left_right/add:z:0:sequential/random_rotation/rotation_matrix/concat:output:0;sequential/random_rotation/transform/strided_slice:output:08sequential/random_rotation/transform/fill_value:output:0*1
_output_shapes
:         ┤┤*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEARо
sequential/random_zoom/ShapeShapeTsequential/random_rotation/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
::э╧t
*sequential/random_zoom/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential/random_zoom/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential/random_zoom/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:─
$sequential/random_zoom/strided_sliceStridedSlice%sequential/random_zoom/Shape:output:03sequential/random_zoom/strided_slice/stack:output:05sequential/random_zoom/strided_slice/stack_1:output:05sequential/random_zoom/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
,sequential/random_zoom/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
¤        Б
.sequential/random_zoom/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
■        x
.sequential/random_zoom/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╠
&sequential/random_zoom/strided_slice_1StridedSlice%sequential/random_zoom/Shape:output:05sequential/random_zoom/strided_slice_1/stack:output:07sequential/random_zoom/strided_slice_1/stack_1:output:07sequential/random_zoom/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskД
sequential/random_zoom/CastCast/sequential/random_zoom/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 
,sequential/random_zoom/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        Б
.sequential/random_zoom/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         x
.sequential/random_zoom/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╠
&sequential/random_zoom/strided_slice_2StridedSlice%sequential/random_zoom/Shape:output:05sequential/random_zoom/strided_slice_2/stack:output:07sequential/random_zoom/strided_slice_2/stack_1:output:07sequential/random_zoom/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЖ
sequential/random_zoom/Cast_1Cast/sequential/random_zoom/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: q
/sequential/random_zoom/stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :╠
-sequential/random_zoom/stateful_uniform/shapePack-sequential/random_zoom/strided_slice:output:08sequential/random_zoom/stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:p
+sequential/random_zoom/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *fff?p
+sequential/random_zoom/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *═╠М?w
-sequential/random_zoom/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ┼
,sequential/random_zoom/stateful_uniform/ProdProd6sequential/random_zoom/stateful_uniform/shape:output:06sequential/random_zoom/stateful_uniform/Const:output:0*
T0*
_output_shapes
: p
.sequential/random_zoom/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :Э
.sequential/random_zoom/stateful_uniform/Cast_1Cast5sequential/random_zoom/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: Т
6sequential/random_zoom/stateful_uniform/RngReadAndSkipRngReadAndSkip?sequential_random_zoom_stateful_uniform_rngreadandskip_resource7sequential/random_zoom/stateful_uniform/Cast/x:output:02sequential/random_zoom/stateful_uniform/Cast_1:y:0*
_output_shapes
:Е
;sequential/random_zoom/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: З
=sequential/random_zoom/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:З
=sequential/random_zoom/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
5sequential/random_zoom/stateful_uniform/strided_sliceStridedSlice>sequential/random_zoom/stateful_uniform/RngReadAndSkip:value:0Dsequential/random_zoom/stateful_uniform/strided_slice/stack:output:0Fsequential/random_zoom/stateful_uniform/strided_slice/stack_1:output:0Fsequential/random_zoom/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskл
/sequential/random_zoom/stateful_uniform/BitcastBitcast>sequential/random_zoom/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0З
=sequential/random_zoom/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Й
?sequential/random_zoom/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Й
?sequential/random_zoom/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Х
7sequential/random_zoom/stateful_uniform/strided_slice_1StridedSlice>sequential/random_zoom/stateful_uniform/RngReadAndSkip:value:0Fsequential/random_zoom/stateful_uniform/strided_slice_1/stack:output:0Hsequential/random_zoom/stateful_uniform/strided_slice_1/stack_1:output:0Hsequential/random_zoom/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:п
1sequential/random_zoom/stateful_uniform/Bitcast_1Bitcast@sequential/random_zoom/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0Ж
Dsequential/random_zoom/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :В
@sequential/random_zoom/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV26sequential/random_zoom/stateful_uniform/shape:output:0:sequential/random_zoom/stateful_uniform/Bitcast_1:output:08sequential/random_zoom/stateful_uniform/Bitcast:output:0Msequential/random_zoom/stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:         ┐
+sequential/random_zoom/stateful_uniform/subSub4sequential/random_zoom/stateful_uniform/max:output:04sequential/random_zoom/stateful_uniform/min:output:0*
T0*
_output_shapes
: р
+sequential/random_zoom/stateful_uniform/mulMulIsequential/random_zoom/stateful_uniform/StatelessRandomUniformV2:output:0/sequential/random_zoom/stateful_uniform/sub:z:0*
T0*'
_output_shapes
:         ╔
'sequential/random_zoom/stateful_uniformAddV2/sequential/random_zoom/stateful_uniform/mul:z:04sequential/random_zoom/stateful_uniform/min:output:0*
T0*'
_output_shapes
:         d
"sequential/random_zoom/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ы
sequential/random_zoom/concatConcatV2+sequential/random_zoom/stateful_uniform:z:0+sequential/random_zoom/stateful_uniform:z:0+sequential/random_zoom/concat/axis:output:0*
N*
T0*'
_output_shapes
:         М
(sequential/random_zoom/zoom_matrix/ShapeShape&sequential/random_zoom/concat:output:0*
T0*
_output_shapes
::э╧А
6sequential/random_zoom/zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: В
8sequential/random_zoom/zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:В
8sequential/random_zoom/zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:А
0sequential/random_zoom/zoom_matrix/strided_sliceStridedSlice1sequential/random_zoom/zoom_matrix/Shape:output:0?sequential/random_zoom/zoom_matrix/strided_slice/stack:output:0Asequential/random_zoom/zoom_matrix/strided_slice/stack_1:output:0Asequential/random_zoom/zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
(sequential/random_zoom/zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?д
&sequential/random_zoom/zoom_matrix/subSub!sequential/random_zoom/Cast_1:y:01sequential/random_zoom/zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: q
,sequential/random_zoom/zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @╣
*sequential/random_zoom/zoom_matrix/truedivRealDiv*sequential/random_zoom/zoom_matrix/sub:z:05sequential/random_zoom/zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: Н
8sequential/random_zoom/zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            П
:sequential/random_zoom/zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           П
:sequential/random_zoom/zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ┼
2sequential/random_zoom/zoom_matrix/strided_slice_1StridedSlice&sequential/random_zoom/concat:output:0Asequential/random_zoom/zoom_matrix/strided_slice_1/stack:output:0Csequential/random_zoom/zoom_matrix/strided_slice_1/stack_1:output:0Csequential/random_zoom/zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_masko
*sequential/random_zoom/zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╙
(sequential/random_zoom/zoom_matrix/sub_1Sub3sequential/random_zoom/zoom_matrix/sub_1/x:output:0;sequential/random_zoom/zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:         ╜
&sequential/random_zoom/zoom_matrix/mulMul.sequential/random_zoom/zoom_matrix/truediv:z:0,sequential/random_zoom/zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:         o
*sequential/random_zoom/zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?ж
(sequential/random_zoom/zoom_matrix/sub_2Subsequential/random_zoom/Cast:y:03sequential/random_zoom/zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: s
.sequential/random_zoom/zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @┐
,sequential/random_zoom/zoom_matrix/truediv_1RealDiv,sequential/random_zoom/zoom_matrix/sub_2:z:07sequential/random_zoom/zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: Н
8sequential/random_zoom/zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           П
:sequential/random_zoom/zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           П
:sequential/random_zoom/zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ┼
2sequential/random_zoom/zoom_matrix/strided_slice_2StridedSlice&sequential/random_zoom/concat:output:0Asequential/random_zoom/zoom_matrix/strided_slice_2/stack:output:0Csequential/random_zoom/zoom_matrix/strided_slice_2/stack_1:output:0Csequential/random_zoom/zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_masko
*sequential/random_zoom/zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╙
(sequential/random_zoom/zoom_matrix/sub_3Sub3sequential/random_zoom/zoom_matrix/sub_3/x:output:0;sequential/random_zoom/zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:         ┴
(sequential/random_zoom/zoom_matrix/mul_1Mul0sequential/random_zoom/zoom_matrix/truediv_1:z:0,sequential/random_zoom/zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:         Н
8sequential/random_zoom/zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            П
:sequential/random_zoom/zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           П
:sequential/random_zoom/zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ┼
2sequential/random_zoom/zoom_matrix/strided_slice_3StridedSlice&sequential/random_zoom/concat:output:0Asequential/random_zoom/zoom_matrix/strided_slice_3/stack:output:0Csequential/random_zoom/zoom_matrix/strided_slice_3/stack_1:output:0Csequential/random_zoom/zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_masks
1sequential/random_zoom/zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :▄
/sequential/random_zoom/zoom_matrix/zeros/packedPack9sequential/random_zoom/zoom_matrix/strided_slice:output:0:sequential/random_zoom/zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:s
.sequential/random_zoom/zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ╒
(sequential/random_zoom/zoom_matrix/zerosFill8sequential/random_zoom/zoom_matrix/zeros/packed:output:07sequential/random_zoom/zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:         u
3sequential/random_zoom/zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :р
1sequential/random_zoom/zoom_matrix/zeros_1/packedPack9sequential/random_zoom/zoom_matrix/strided_slice:output:0<sequential/random_zoom/zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:u
0sequential/random_zoom/zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    █
*sequential/random_zoom/zoom_matrix/zeros_1Fill:sequential/random_zoom/zoom_matrix/zeros_1/packed:output:09sequential/random_zoom/zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:         Н
8sequential/random_zoom/zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           П
:sequential/random_zoom/zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           П
:sequential/random_zoom/zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ┼
2sequential/random_zoom/zoom_matrix/strided_slice_4StridedSlice&sequential/random_zoom/concat:output:0Asequential/random_zoom/zoom_matrix/strided_slice_4/stack:output:0Csequential/random_zoom/zoom_matrix/strided_slice_4/stack_1:output:0Csequential/random_zoom/zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_masku
3sequential/random_zoom/zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :р
1sequential/random_zoom/zoom_matrix/zeros_2/packedPack9sequential/random_zoom/zoom_matrix/strided_slice:output:0<sequential/random_zoom/zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:u
0sequential/random_zoom/zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    █
*sequential/random_zoom/zoom_matrix/zeros_2Fill:sequential/random_zoom/zoom_matrix/zeros_2/packed:output:09sequential/random_zoom/zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:         p
.sequential/random_zoom/zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ъ
)sequential/random_zoom/zoom_matrix/concatConcatV2;sequential/random_zoom/zoom_matrix/strided_slice_3:output:01sequential/random_zoom/zoom_matrix/zeros:output:0*sequential/random_zoom/zoom_matrix/mul:z:03sequential/random_zoom/zoom_matrix/zeros_1:output:0;sequential/random_zoom/zoom_matrix/strided_slice_4:output:0,sequential/random_zoom/zoom_matrix/mul_1:z:03sequential/random_zoom/zoom_matrix/zeros_2:output:07sequential/random_zoom/zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:         ╕
&sequential/random_zoom/transform/ShapeShapeTsequential/random_rotation/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
::э╧~
4sequential/random_zoom/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:А
6sequential/random_zoom/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:А
6sequential/random_zoom/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:т
.sequential/random_zoom/transform/strided_sliceStridedSlice/sequential/random_zoom/transform/Shape:output:0=sequential/random_zoom/transform/strided_slice/stack:output:0?sequential/random_zoom/transform/strided_slice/stack_1:output:0?sequential/random_zoom/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:p
+sequential/random_zoom/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ╟
;sequential/random_zoom/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3Tsequential/random_rotation/transform/ImageProjectiveTransformV3:transformed_images:02sequential/random_zoom/zoom_matrix/concat:output:07sequential/random_zoom/transform/strided_slice:output:04sequential/random_zoom/transform/fill_value:output:0*1
_output_shapes
:         ┤┤*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEARй
IdentityIdentityPsequential/random_zoom/transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:         ┤┤■
NoOpNoOp@^sequential/random_flip/stateful_uniform_full_int/RngReadAndSkip;^sequential/random_rotation/stateful_uniform/RngReadAndSkip7^sequential/random_zoom/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         ┤┤: : : 2В
?sequential/random_flip/stateful_uniform_full_int/RngReadAndSkip?sequential/random_flip/stateful_uniform_full_int/RngReadAndSkip2x
:sequential/random_rotation/stateful_uniform/RngReadAndSkip:sequential/random_rotation/stateful_uniform/RngReadAndSkip2p
6sequential/random_zoom/stateful_uniform/RngReadAndSkip6sequential/random_zoom/stateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameargs_0
┬
Х
$__inference_dense_layer_call_fn_9300

inputs
unknown:АЄА
	unknown_0:	А
identityИвStatefulPartitionedCall╒
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_7375p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         АЄ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:         АЄ
 
_user_specified_nameinputs
У
a
E__inference_random_flip_layer_call_and_return_conditional_losses_9694

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:         ┤┤"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ┤┤:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameinputs
Л
∙
@__inference_conv2d_layer_call_and_return_conditional_losses_9213

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ы
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ┤┤*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ┤┤Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:         ┤┤k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:         ┤┤w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ┤┤: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameinputs
∙з
Щ
H__inference_module_wrapper_layer_call_and_return_conditional_losses_8761

args_0V
Hsequential_random_flip_stateful_uniform_full_int_rngreadandskip_resource:	Q
Csequential_random_rotation_stateful_uniform_rngreadandskip_resource:	M
?sequential_random_zoom_stateful_uniform_rngreadandskip_resource:	
identityИв?sequential/random_flip/stateful_uniform_full_int/RngReadAndSkipв:sequential/random_rotation/stateful_uniform/RngReadAndSkipв6sequential/random_zoom/stateful_uniform/RngReadAndSkipА
6sequential/random_flip/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:А
6sequential/random_flip/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: р
5sequential/random_flip/stateful_uniform_full_int/ProdProd?sequential/random_flip/stateful_uniform_full_int/shape:output:0?sequential/random_flip/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: y
7sequential/random_flip/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :п
7sequential/random_flip/stateful_uniform_full_int/Cast_1Cast>sequential/random_flip/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ╢
?sequential/random_flip/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkipHsequential_random_flip_stateful_uniform_full_int_rngreadandskip_resource@sequential/random_flip/stateful_uniform_full_int/Cast/x:output:0;sequential/random_flip/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:О
Dsequential/random_flip/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Р
Fsequential/random_flip/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Р
Fsequential/random_flip/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╠
>sequential/random_flip/stateful_uniform_full_int/strided_sliceStridedSliceGsequential/random_flip/stateful_uniform_full_int/RngReadAndSkip:value:0Msequential/random_flip/stateful_uniform_full_int/strided_slice/stack:output:0Osequential/random_flip/stateful_uniform_full_int/strided_slice/stack_1:output:0Osequential/random_flip/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask╜
8sequential/random_flip/stateful_uniform_full_int/BitcastBitcastGsequential/random_flip/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0Р
Fsequential/random_flip/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Т
Hsequential/random_flip/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Т
Hsequential/random_flip/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┬
@sequential/random_flip/stateful_uniform_full_int/strided_slice_1StridedSliceGsequential/random_flip/stateful_uniform_full_int/RngReadAndSkip:value:0Osequential/random_flip/stateful_uniform_full_int/strided_slice_1/stack:output:0Qsequential/random_flip/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Qsequential/random_flip/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:┴
:sequential/random_flip/stateful_uniform_full_int/Bitcast_1BitcastIsequential/random_flip/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0v
4sequential/random_flip/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :Д
0sequential/random_flip/stateful_uniform_full_intStatelessRandomUniformFullIntV2?sequential/random_flip/stateful_uniform_full_int/shape:output:0Csequential/random_flip/stateful_uniform_full_int/Bitcast_1:output:0Asequential/random_flip/stateful_uniform_full_int/Bitcast:output:0=sequential/random_flip/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	k
!sequential/random_flip/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R ╜
sequential/random_flip/stackPack9sequential/random_flip/stateful_uniform_full_int:output:0*sequential/random_flip/zeros_like:output:0*
N*
T0	*
_output_shapes

:{
*sequential/random_flip/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        }
,sequential/random_flip/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,sequential/random_flip/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ъ
$sequential/random_flip/strided_sliceStridedSlice%sequential/random_flip/stack:output:03sequential/random_flip/strided_slice/stack:output:05sequential/random_flip/strided_slice/stack_1:output:05sequential/random_flip/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask╡
Jsequential/random_flip/stateless_random_flip_left_right/control_dependencyIdentityargs_0*
T0*
_class
loc:@args_0*1
_output_shapes
:         ┤┤╬
=sequential/random_flip/stateless_random_flip_left_right/ShapeShapeSsequential/random_flip/stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
::э╧Х
Ksequential/random_flip/stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ч
Msequential/random_flip/stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ч
Msequential/random_flip/stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
Esequential/random_flip/stateless_random_flip_left_right/strided_sliceStridedSliceFsequential/random_flip/stateless_random_flip_left_right/Shape:output:0Tsequential/random_flip/stateless_random_flip_left_right/strided_slice/stack:output:0Vsequential/random_flip/stateless_random_flip_left_right/strided_slice/stack_1:output:0Vsequential/random_flip/stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask▄
Vsequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/shapePackNsequential/random_flip/stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:Щ
Tsequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    Щ
Tsequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  А?▐
msequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter-sequential/random_flip/strided_slice:output:0* 
_output_shapes
::п
msequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :ё
isequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2_sequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/shape:output:0ssequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0wsequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0vsequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:         ║
Tsequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/subSub]sequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/max:output:0]sequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ╫
Tsequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/mulMulrsequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Xsequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:         └
Psequential/random_flip/stateless_random_flip_left_right/stateless_random_uniformAddV2Xsequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/mul:z:0]sequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:         Й
Gsequential/random_flip/stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Й
Gsequential/random_flip/stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Й
Gsequential/random_flip/stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :┴
Esequential/random_flip/stateless_random_flip_left_right/Reshape/shapePackNsequential/random_flip/stateless_random_flip_left_right/strided_slice:output:0Psequential/random_flip/stateless_random_flip_left_right/Reshape/shape/1:output:0Psequential/random_flip/stateless_random_flip_left_right/Reshape/shape/2:output:0Psequential/random_flip/stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:к
?sequential/random_flip/stateless_random_flip_left_right/ReshapeReshapeTsequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform:z:0Nsequential/random_flip/stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:         ╩
=sequential/random_flip/stateless_random_flip_left_right/RoundRoundHsequential/random_flip/stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:         Р
Fsequential/random_flip/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:░
Asequential/random_flip/stateless_random_flip_left_right/ReverseV2	ReverseV2Ssequential/random_flip/stateless_random_flip_left_right/control_dependency:output:0Osequential/random_flip/stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:         ┤┤Н
;sequential/random_flip/stateless_random_flip_left_right/mulMulAsequential/random_flip/stateless_random_flip_left_right/Round:y:0Jsequential/random_flip/stateless_random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:         ┤┤В
=sequential/random_flip/stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?З
;sequential/random_flip/stateless_random_flip_left_right/subSubFsequential/random_flip/stateless_random_flip_left_right/sub/x:output:0Asequential/random_flip/stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:         Ц
=sequential/random_flip/stateless_random_flip_left_right/mul_1Mul?sequential/random_flip/stateless_random_flip_left_right/sub:z:0Ssequential/random_flip/stateless_random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:         ┤┤Д
;sequential/random_flip/stateless_random_flip_left_right/addAddV2?sequential/random_flip/stateless_random_flip_left_right/mul:z:0Asequential/random_flip/stateless_random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:         ┤┤Э
 sequential/random_rotation/ShapeShape?sequential/random_flip/stateless_random_flip_left_right/add:z:0*
T0*
_output_shapes
::э╧x
.sequential/random_rotation/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0sequential/random_rotation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0sequential/random_rotation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╪
(sequential/random_rotation/strided_sliceStridedSlice)sequential/random_rotation/Shape:output:07sequential/random_rotation/strided_slice/stack:output:09sequential/random_rotation/strided_slice/stack_1:output:09sequential/random_rotation/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskГ
0sequential/random_rotation/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
¤        Е
2sequential/random_rotation/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
■        |
2sequential/random_rotation/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:р
*sequential/random_rotation/strided_slice_1StridedSlice)sequential/random_rotation/Shape:output:09sequential/random_rotation/strided_slice_1/stack:output:0;sequential/random_rotation/strided_slice_1/stack_1:output:0;sequential/random_rotation/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskМ
sequential/random_rotation/CastCast3sequential/random_rotation/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: Г
0sequential/random_rotation/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        Е
2sequential/random_rotation/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         |
2sequential/random_rotation/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:р
*sequential/random_rotation/strided_slice_2StridedSlice)sequential/random_rotation/Shape:output:09sequential/random_rotation/strided_slice_2/stack:output:0;sequential/random_rotation/strided_slice_2/stack_1:output:0;sequential/random_rotation/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskО
!sequential/random_rotation/Cast_1Cast3sequential/random_rotation/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: Ъ
1sequential/random_rotation/stateful_uniform/shapePack1sequential/random_rotation/strided_slice:output:0*
N*
T0*
_output_shapes
:t
/sequential/random_rotation/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|┘ ┐t
/sequential/random_rotation/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|┘ ?{
1sequential/random_rotation/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ╤
0sequential/random_rotation/stateful_uniform/ProdProd:sequential/random_rotation/stateful_uniform/shape:output:0:sequential/random_rotation/stateful_uniform/Const:output:0*
T0*
_output_shapes
: t
2sequential/random_rotation/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :е
2sequential/random_rotation/stateful_uniform/Cast_1Cast9sequential/random_rotation/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: в
:sequential/random_rotation/stateful_uniform/RngReadAndSkipRngReadAndSkipCsequential_random_rotation_stateful_uniform_rngreadandskip_resource;sequential/random_rotation/stateful_uniform/Cast/x:output:06sequential/random_rotation/stateful_uniform/Cast_1:y:0*
_output_shapes
:Й
?sequential/random_rotation/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Л
Asequential/random_rotation/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Л
Asequential/random_rotation/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:│
9sequential/random_rotation/stateful_uniform/strided_sliceStridedSliceBsequential/random_rotation/stateful_uniform/RngReadAndSkip:value:0Hsequential/random_rotation/stateful_uniform/strided_slice/stack:output:0Jsequential/random_rotation/stateful_uniform/strided_slice/stack_1:output:0Jsequential/random_rotation/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask│
3sequential/random_rotation/stateful_uniform/BitcastBitcastBsequential/random_rotation/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0Л
Asequential/random_rotation/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Н
Csequential/random_rotation/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Н
Csequential/random_rotation/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
;sequential/random_rotation/stateful_uniform/strided_slice_1StridedSliceBsequential/random_rotation/stateful_uniform/RngReadAndSkip:value:0Jsequential/random_rotation/stateful_uniform/strided_slice_1/stack:output:0Lsequential/random_rotation/stateful_uniform/strided_slice_1/stack_1:output:0Lsequential/random_rotation/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:╖
5sequential/random_rotation/stateful_uniform/Bitcast_1BitcastDsequential/random_rotation/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0К
Hsequential/random_rotation/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :Т
Dsequential/random_rotation/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2:sequential/random_rotation/stateful_uniform/shape:output:0>sequential/random_rotation/stateful_uniform/Bitcast_1:output:0<sequential/random_rotation/stateful_uniform/Bitcast:output:0Qsequential/random_rotation/stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:         ╦
/sequential/random_rotation/stateful_uniform/subSub8sequential/random_rotation/stateful_uniform/max:output:08sequential/random_rotation/stateful_uniform/min:output:0*
T0*
_output_shapes
: ш
/sequential/random_rotation/stateful_uniform/mulMulMsequential/random_rotation/stateful_uniform/StatelessRandomUniformV2:output:03sequential/random_rotation/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:         ╤
+sequential/random_rotation/stateful_uniformAddV23sequential/random_rotation/stateful_uniform/mul:z:08sequential/random_rotation/stateful_uniform/min:output:0*
T0*#
_output_shapes
:         u
0sequential/random_rotation/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╕
.sequential/random_rotation/rotation_matrix/subSub%sequential/random_rotation/Cast_1:y:09sequential/random_rotation/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: Ф
.sequential/random_rotation/rotation_matrix/CosCos/sequential/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:         w
2sequential/random_rotation/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╝
0sequential/random_rotation/rotation_matrix/sub_1Sub%sequential/random_rotation/Cast_1:y:0;sequential/random_rotation/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: ═
.sequential/random_rotation/rotation_matrix/mulMul2sequential/random_rotation/rotation_matrix/Cos:y:04sequential/random_rotation/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:         Ф
.sequential/random_rotation/rotation_matrix/SinSin/sequential/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:         w
2sequential/random_rotation/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?║
0sequential/random_rotation/rotation_matrix/sub_2Sub#sequential/random_rotation/Cast:y:0;sequential/random_rotation/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: ╧
0sequential/random_rotation/rotation_matrix/mul_1Mul2sequential/random_rotation/rotation_matrix/Sin:y:04sequential/random_rotation/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:         ╧
0sequential/random_rotation/rotation_matrix/sub_3Sub2sequential/random_rotation/rotation_matrix/mul:z:04sequential/random_rotation/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:         ╧
0sequential/random_rotation/rotation_matrix/sub_4Sub2sequential/random_rotation/rotation_matrix/sub:z:04sequential/random_rotation/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:         y
4sequential/random_rotation/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @р
2sequential/random_rotation/rotation_matrix/truedivRealDiv4sequential/random_rotation/rotation_matrix/sub_4:z:0=sequential/random_rotation/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:         w
2sequential/random_rotation/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?║
0sequential/random_rotation/rotation_matrix/sub_5Sub#sequential/random_rotation/Cast:y:0;sequential/random_rotation/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: Ц
0sequential/random_rotation/rotation_matrix/Sin_1Sin/sequential/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:         w
2sequential/random_rotation/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╝
0sequential/random_rotation/rotation_matrix/sub_6Sub%sequential/random_rotation/Cast_1:y:0;sequential/random_rotation/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: ╤
0sequential/random_rotation/rotation_matrix/mul_2Mul4sequential/random_rotation/rotation_matrix/Sin_1:y:04sequential/random_rotation/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:         Ц
0sequential/random_rotation/rotation_matrix/Cos_1Cos/sequential/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:         w
2sequential/random_rotation/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?║
0sequential/random_rotation/rotation_matrix/sub_7Sub#sequential/random_rotation/Cast:y:0;sequential/random_rotation/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: ╤
0sequential/random_rotation/rotation_matrix/mul_3Mul4sequential/random_rotation/rotation_matrix/Cos_1:y:04sequential/random_rotation/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:         ╤
.sequential/random_rotation/rotation_matrix/addAddV24sequential/random_rotation/rotation_matrix/mul_2:z:04sequential/random_rotation/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:         ╧
0sequential/random_rotation/rotation_matrix/sub_8Sub4sequential/random_rotation/rotation_matrix/sub_5:z:02sequential/random_rotation/rotation_matrix/add:z:0*
T0*#
_output_shapes
:         {
6sequential/random_rotation/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @ф
4sequential/random_rotation/rotation_matrix/truediv_1RealDiv4sequential/random_rotation/rotation_matrix/sub_8:z:0?sequential/random_rotation/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:         Э
0sequential/random_rotation/rotation_matrix/ShapeShape/sequential/random_rotation/stateful_uniform:z:0*
T0*
_output_shapes
::э╧И
>sequential/random_rotation/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: К
@sequential/random_rotation/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:К
@sequential/random_rotation/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:и
8sequential/random_rotation/rotation_matrix/strided_sliceStridedSlice9sequential/random_rotation/rotation_matrix/Shape:output:0Gsequential/random_rotation/rotation_matrix/strided_slice/stack:output:0Isequential/random_rotation/rotation_matrix/strided_slice/stack_1:output:0Isequential/random_rotation/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЦ
0sequential/random_rotation/rotation_matrix/Cos_2Cos/sequential/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:         С
@sequential/random_rotation/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        У
Bsequential/random_rotation/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        У
Bsequential/random_rotation/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      █
:sequential/random_rotation/rotation_matrix/strided_slice_1StridedSlice4sequential/random_rotation/rotation_matrix/Cos_2:y:0Isequential/random_rotation/rotation_matrix/strided_slice_1/stack:output:0Ksequential/random_rotation/rotation_matrix/strided_slice_1/stack_1:output:0Ksequential/random_rotation/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_maskЦ
0sequential/random_rotation/rotation_matrix/Sin_2Sin/sequential/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:         С
@sequential/random_rotation/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        У
Bsequential/random_rotation/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        У
Bsequential/random_rotation/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      █
:sequential/random_rotation/rotation_matrix/strided_slice_2StridedSlice4sequential/random_rotation/rotation_matrix/Sin_2:y:0Isequential/random_rotation/rotation_matrix/strided_slice_2/stack:output:0Ksequential/random_rotation/rotation_matrix/strided_slice_2/stack_1:output:0Ksequential/random_rotation/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_maskм
.sequential/random_rotation/rotation_matrix/NegNegCsequential/random_rotation/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:         С
@sequential/random_rotation/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        У
Bsequential/random_rotation/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        У
Bsequential/random_rotation/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ▌
:sequential/random_rotation/rotation_matrix/strided_slice_3StridedSlice6sequential/random_rotation/rotation_matrix/truediv:z:0Isequential/random_rotation/rotation_matrix/strided_slice_3/stack:output:0Ksequential/random_rotation/rotation_matrix/strided_slice_3/stack_1:output:0Ksequential/random_rotation/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_maskЦ
0sequential/random_rotation/rotation_matrix/Sin_3Sin/sequential/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:         С
@sequential/random_rotation/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        У
Bsequential/random_rotation/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        У
Bsequential/random_rotation/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      █
:sequential/random_rotation/rotation_matrix/strided_slice_4StridedSlice4sequential/random_rotation/rotation_matrix/Sin_3:y:0Isequential/random_rotation/rotation_matrix/strided_slice_4/stack:output:0Ksequential/random_rotation/rotation_matrix/strided_slice_4/stack_1:output:0Ksequential/random_rotation/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_maskЦ
0sequential/random_rotation/rotation_matrix/Cos_3Cos/sequential/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:         С
@sequential/random_rotation/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        У
Bsequential/random_rotation/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        У
Bsequential/random_rotation/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      █
:sequential/random_rotation/rotation_matrix/strided_slice_5StridedSlice4sequential/random_rotation/rotation_matrix/Cos_3:y:0Isequential/random_rotation/rotation_matrix/strided_slice_5/stack:output:0Ksequential/random_rotation/rotation_matrix/strided_slice_5/stack_1:output:0Ksequential/random_rotation/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_maskС
@sequential/random_rotation/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        У
Bsequential/random_rotation/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        У
Bsequential/random_rotation/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ▀
:sequential/random_rotation/rotation_matrix/strided_slice_6StridedSlice8sequential/random_rotation/rotation_matrix/truediv_1:z:0Isequential/random_rotation/rotation_matrix/strided_slice_6/stack:output:0Ksequential/random_rotation/rotation_matrix/strided_slice_6/stack_1:output:0Ksequential/random_rotation/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask{
9sequential/random_rotation/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ї
7sequential/random_rotation/rotation_matrix/zeros/packedPackAsequential/random_rotation/rotation_matrix/strided_slice:output:0Bsequential/random_rotation/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:{
6sequential/random_rotation/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    э
0sequential/random_rotation/rotation_matrix/zerosFill@sequential/random_rotation/rotation_matrix/zeros/packed:output:0?sequential/random_rotation/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:         x
6sequential/random_rotation/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Б
1sequential/random_rotation/rotation_matrix/concatConcatV2Csequential/random_rotation/rotation_matrix/strided_slice_1:output:02sequential/random_rotation/rotation_matrix/Neg:y:0Csequential/random_rotation/rotation_matrix/strided_slice_3:output:0Csequential/random_rotation/rotation_matrix/strided_slice_4:output:0Csequential/random_rotation/rotation_matrix/strided_slice_5:output:0Csequential/random_rotation/rotation_matrix/strided_slice_6:output:09sequential/random_rotation/rotation_matrix/zeros:output:0?sequential/random_rotation/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:         з
*sequential/random_rotation/transform/ShapeShape?sequential/random_flip/stateless_random_flip_left_right/add:z:0*
T0*
_output_shapes
::э╧В
8sequential/random_rotation/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:Д
:sequential/random_rotation/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Д
:sequential/random_rotation/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ў
2sequential/random_rotation/transform/strided_sliceStridedSlice3sequential/random_rotation/transform/Shape:output:0Asequential/random_rotation/transform/strided_slice/stack:output:0Csequential/random_rotation/transform/strided_slice/stack_1:output:0Csequential/random_rotation/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:t
/sequential/random_rotation/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ╞
?sequential/random_rotation/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3?sequential/random_flip/stateless_random_flip_left_right/add:z:0:sequential/random_rotation/rotation_matrix/concat:output:0;sequential/random_rotation/transform/strided_slice:output:08sequential/random_rotation/transform/fill_value:output:0*1
_output_shapes
:         ┤┤*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEARо
sequential/random_zoom/ShapeShapeTsequential/random_rotation/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
::э╧t
*sequential/random_zoom/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential/random_zoom/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential/random_zoom/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:─
$sequential/random_zoom/strided_sliceStridedSlice%sequential/random_zoom/Shape:output:03sequential/random_zoom/strided_slice/stack:output:05sequential/random_zoom/strided_slice/stack_1:output:05sequential/random_zoom/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
,sequential/random_zoom/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
¤        Б
.sequential/random_zoom/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
■        x
.sequential/random_zoom/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╠
&sequential/random_zoom/strided_slice_1StridedSlice%sequential/random_zoom/Shape:output:05sequential/random_zoom/strided_slice_1/stack:output:07sequential/random_zoom/strided_slice_1/stack_1:output:07sequential/random_zoom/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskД
sequential/random_zoom/CastCast/sequential/random_zoom/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 
,sequential/random_zoom/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        Б
.sequential/random_zoom/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         x
.sequential/random_zoom/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╠
&sequential/random_zoom/strided_slice_2StridedSlice%sequential/random_zoom/Shape:output:05sequential/random_zoom/strided_slice_2/stack:output:07sequential/random_zoom/strided_slice_2/stack_1:output:07sequential/random_zoom/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЖ
sequential/random_zoom/Cast_1Cast/sequential/random_zoom/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: q
/sequential/random_zoom/stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :╠
-sequential/random_zoom/stateful_uniform/shapePack-sequential/random_zoom/strided_slice:output:08sequential/random_zoom/stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:p
+sequential/random_zoom/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *fff?p
+sequential/random_zoom/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *═╠М?w
-sequential/random_zoom/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ┼
,sequential/random_zoom/stateful_uniform/ProdProd6sequential/random_zoom/stateful_uniform/shape:output:06sequential/random_zoom/stateful_uniform/Const:output:0*
T0*
_output_shapes
: p
.sequential/random_zoom/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :Э
.sequential/random_zoom/stateful_uniform/Cast_1Cast5sequential/random_zoom/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: Т
6sequential/random_zoom/stateful_uniform/RngReadAndSkipRngReadAndSkip?sequential_random_zoom_stateful_uniform_rngreadandskip_resource7sequential/random_zoom/stateful_uniform/Cast/x:output:02sequential/random_zoom/stateful_uniform/Cast_1:y:0*
_output_shapes
:Е
;sequential/random_zoom/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: З
=sequential/random_zoom/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:З
=sequential/random_zoom/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
5sequential/random_zoom/stateful_uniform/strided_sliceStridedSlice>sequential/random_zoom/stateful_uniform/RngReadAndSkip:value:0Dsequential/random_zoom/stateful_uniform/strided_slice/stack:output:0Fsequential/random_zoom/stateful_uniform/strided_slice/stack_1:output:0Fsequential/random_zoom/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskл
/sequential/random_zoom/stateful_uniform/BitcastBitcast>sequential/random_zoom/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0З
=sequential/random_zoom/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Й
?sequential/random_zoom/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Й
?sequential/random_zoom/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Х
7sequential/random_zoom/stateful_uniform/strided_slice_1StridedSlice>sequential/random_zoom/stateful_uniform/RngReadAndSkip:value:0Fsequential/random_zoom/stateful_uniform/strided_slice_1/stack:output:0Hsequential/random_zoom/stateful_uniform/strided_slice_1/stack_1:output:0Hsequential/random_zoom/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:п
1sequential/random_zoom/stateful_uniform/Bitcast_1Bitcast@sequential/random_zoom/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0Ж
Dsequential/random_zoom/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :В
@sequential/random_zoom/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV26sequential/random_zoom/stateful_uniform/shape:output:0:sequential/random_zoom/stateful_uniform/Bitcast_1:output:08sequential/random_zoom/stateful_uniform/Bitcast:output:0Msequential/random_zoom/stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:         ┐
+sequential/random_zoom/stateful_uniform/subSub4sequential/random_zoom/stateful_uniform/max:output:04sequential/random_zoom/stateful_uniform/min:output:0*
T0*
_output_shapes
: р
+sequential/random_zoom/stateful_uniform/mulMulIsequential/random_zoom/stateful_uniform/StatelessRandomUniformV2:output:0/sequential/random_zoom/stateful_uniform/sub:z:0*
T0*'
_output_shapes
:         ╔
'sequential/random_zoom/stateful_uniformAddV2/sequential/random_zoom/stateful_uniform/mul:z:04sequential/random_zoom/stateful_uniform/min:output:0*
T0*'
_output_shapes
:         d
"sequential/random_zoom/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ы
sequential/random_zoom/concatConcatV2+sequential/random_zoom/stateful_uniform:z:0+sequential/random_zoom/stateful_uniform:z:0+sequential/random_zoom/concat/axis:output:0*
N*
T0*'
_output_shapes
:         М
(sequential/random_zoom/zoom_matrix/ShapeShape&sequential/random_zoom/concat:output:0*
T0*
_output_shapes
::э╧А
6sequential/random_zoom/zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: В
8sequential/random_zoom/zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:В
8sequential/random_zoom/zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:А
0sequential/random_zoom/zoom_matrix/strided_sliceStridedSlice1sequential/random_zoom/zoom_matrix/Shape:output:0?sequential/random_zoom/zoom_matrix/strided_slice/stack:output:0Asequential/random_zoom/zoom_matrix/strided_slice/stack_1:output:0Asequential/random_zoom/zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
(sequential/random_zoom/zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?д
&sequential/random_zoom/zoom_matrix/subSub!sequential/random_zoom/Cast_1:y:01sequential/random_zoom/zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: q
,sequential/random_zoom/zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @╣
*sequential/random_zoom/zoom_matrix/truedivRealDiv*sequential/random_zoom/zoom_matrix/sub:z:05sequential/random_zoom/zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: Н
8sequential/random_zoom/zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            П
:sequential/random_zoom/zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           П
:sequential/random_zoom/zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ┼
2sequential/random_zoom/zoom_matrix/strided_slice_1StridedSlice&sequential/random_zoom/concat:output:0Asequential/random_zoom/zoom_matrix/strided_slice_1/stack:output:0Csequential/random_zoom/zoom_matrix/strided_slice_1/stack_1:output:0Csequential/random_zoom/zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_masko
*sequential/random_zoom/zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╙
(sequential/random_zoom/zoom_matrix/sub_1Sub3sequential/random_zoom/zoom_matrix/sub_1/x:output:0;sequential/random_zoom/zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:         ╜
&sequential/random_zoom/zoom_matrix/mulMul.sequential/random_zoom/zoom_matrix/truediv:z:0,sequential/random_zoom/zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:         o
*sequential/random_zoom/zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?ж
(sequential/random_zoom/zoom_matrix/sub_2Subsequential/random_zoom/Cast:y:03sequential/random_zoom/zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: s
.sequential/random_zoom/zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @┐
,sequential/random_zoom/zoom_matrix/truediv_1RealDiv,sequential/random_zoom/zoom_matrix/sub_2:z:07sequential/random_zoom/zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: Н
8sequential/random_zoom/zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           П
:sequential/random_zoom/zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           П
:sequential/random_zoom/zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ┼
2sequential/random_zoom/zoom_matrix/strided_slice_2StridedSlice&sequential/random_zoom/concat:output:0Asequential/random_zoom/zoom_matrix/strided_slice_2/stack:output:0Csequential/random_zoom/zoom_matrix/strided_slice_2/stack_1:output:0Csequential/random_zoom/zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_masko
*sequential/random_zoom/zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╙
(sequential/random_zoom/zoom_matrix/sub_3Sub3sequential/random_zoom/zoom_matrix/sub_3/x:output:0;sequential/random_zoom/zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:         ┴
(sequential/random_zoom/zoom_matrix/mul_1Mul0sequential/random_zoom/zoom_matrix/truediv_1:z:0,sequential/random_zoom/zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:         Н
8sequential/random_zoom/zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            П
:sequential/random_zoom/zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           П
:sequential/random_zoom/zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ┼
2sequential/random_zoom/zoom_matrix/strided_slice_3StridedSlice&sequential/random_zoom/concat:output:0Asequential/random_zoom/zoom_matrix/strided_slice_3/stack:output:0Csequential/random_zoom/zoom_matrix/strided_slice_3/stack_1:output:0Csequential/random_zoom/zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_masks
1sequential/random_zoom/zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :▄
/sequential/random_zoom/zoom_matrix/zeros/packedPack9sequential/random_zoom/zoom_matrix/strided_slice:output:0:sequential/random_zoom/zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:s
.sequential/random_zoom/zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ╒
(sequential/random_zoom/zoom_matrix/zerosFill8sequential/random_zoom/zoom_matrix/zeros/packed:output:07sequential/random_zoom/zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:         u
3sequential/random_zoom/zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :р
1sequential/random_zoom/zoom_matrix/zeros_1/packedPack9sequential/random_zoom/zoom_matrix/strided_slice:output:0<sequential/random_zoom/zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:u
0sequential/random_zoom/zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    █
*sequential/random_zoom/zoom_matrix/zeros_1Fill:sequential/random_zoom/zoom_matrix/zeros_1/packed:output:09sequential/random_zoom/zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:         Н
8sequential/random_zoom/zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           П
:sequential/random_zoom/zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           П
:sequential/random_zoom/zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ┼
2sequential/random_zoom/zoom_matrix/strided_slice_4StridedSlice&sequential/random_zoom/concat:output:0Asequential/random_zoom/zoom_matrix/strided_slice_4/stack:output:0Csequential/random_zoom/zoom_matrix/strided_slice_4/stack_1:output:0Csequential/random_zoom/zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_masku
3sequential/random_zoom/zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :р
1sequential/random_zoom/zoom_matrix/zeros_2/packedPack9sequential/random_zoom/zoom_matrix/strided_slice:output:0<sequential/random_zoom/zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:u
0sequential/random_zoom/zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    █
*sequential/random_zoom/zoom_matrix/zeros_2Fill:sequential/random_zoom/zoom_matrix/zeros_2/packed:output:09sequential/random_zoom/zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:         p
.sequential/random_zoom/zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ъ
)sequential/random_zoom/zoom_matrix/concatConcatV2;sequential/random_zoom/zoom_matrix/strided_slice_3:output:01sequential/random_zoom/zoom_matrix/zeros:output:0*sequential/random_zoom/zoom_matrix/mul:z:03sequential/random_zoom/zoom_matrix/zeros_1:output:0;sequential/random_zoom/zoom_matrix/strided_slice_4:output:0,sequential/random_zoom/zoom_matrix/mul_1:z:03sequential/random_zoom/zoom_matrix/zeros_2:output:07sequential/random_zoom/zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:         ╕
&sequential/random_zoom/transform/ShapeShapeTsequential/random_rotation/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
::э╧~
4sequential/random_zoom/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:А
6sequential/random_zoom/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:А
6sequential/random_zoom/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:т
.sequential/random_zoom/transform/strided_sliceStridedSlice/sequential/random_zoom/transform/Shape:output:0=sequential/random_zoom/transform/strided_slice/stack:output:0?sequential/random_zoom/transform/strided_slice/stack_1:output:0?sequential/random_zoom/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:p
+sequential/random_zoom/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ╟
;sequential/random_zoom/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3Tsequential/random_rotation/transform/ImageProjectiveTransformV3:transformed_images:02sequential/random_zoom/zoom_matrix/concat:output:07sequential/random_zoom/transform/strided_slice:output:04sequential/random_zoom/transform/fill_value:output:0*1
_output_shapes
:         ┤┤*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEARй
IdentityIdentityPsequential/random_zoom/transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:         ┤┤■
NoOpNoOp@^sequential/random_flip/stateful_uniform_full_int/RngReadAndSkip;^sequential/random_rotation/stateful_uniform/RngReadAndSkip7^sequential/random_zoom/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         ┤┤: : : 2В
?sequential/random_flip/stateful_uniform_full_int/RngReadAndSkip?sequential/random_flip/stateful_uniform_full_int/RngReadAndSkip2x
:sequential/random_rotation/stateful_uniform/RngReadAndSkip:sequential/random_rotation/stateful_uniform/RngReadAndSkip2p
6sequential/random_zoom/stateful_uniform/RngReadAndSkip6sequential/random_zoom/stateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameargs_0
┴

Т
"__inference_signature_wrapper_8046
module_wrapper_input!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@
	unknown_5:АЄА
	unknown_6:	А
	unknown_7:	А
	unknown_8:
identityИвStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *(
f#R!
__inference__wrapped_model_7236o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:         ┤┤: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
1
_output_shapes
:         ┤┤
.
_user_specified_namemodule_wrapper_input
Б
√
B__inference_conv2d_1_layer_call_and_return_conditional_losses_7324

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ZZ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ZZ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         ZZ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         ZZ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ZZ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         ZZ
 
_user_specified_nameinputs
Т
`
D__inference_sequential_layer_call_and_return_conditional_losses_9620

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:         ┤┤"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ┤┤:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameinputs
Б
√
B__inference_conv2d_1_layer_call_and_return_conditional_losses_9233

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ZZ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ZZ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         ZZ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         ZZ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ZZ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         ZZ
 
_user_specified_nameinputs
┤
J
.__inference_max_pooling2d_1_layer_call_fn_7260

inputs
identity╫
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_7254Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
░
H
,__inference_max_pooling2d_layer_call_fn_7248

inputs
identity╒
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_7242Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╚	
є
A__inference_outputs_layer_call_and_return_conditional_losses_9330

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
┼
E
)__inference_sequential_layer_call_fn_9346

inputs
identity╣
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_9109j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ┤┤"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ┤┤:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameinputs
Р
╚
D__inference_sequential_layer_call_and_return_conditional_losses_9056
random_flip_input
random_flip_8824:	"
random_rotation_8946:	
random_zoom_9052:	
identityИв#random_flip/StatefulPartitionedCallв'random_rotation/StatefulPartitionedCallв#random_zoom/StatefulPartitionedCallЇ
#random_flip/StatefulPartitionedCallStatefulPartitionedCallrandom_flip_inputrandom_flip_8824*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_random_flip_layer_call_and_return_conditional_losses_8823Ы
'random_rotation/StatefulPartitionedCallStatefulPartitionedCall,random_flip/StatefulPartitionedCall:output:0random_rotation_8946*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_random_rotation_layer_call_and_return_conditional_losses_8945У
#random_zoom/StatefulPartitionedCallStatefulPartitionedCall0random_rotation/StatefulPartitionedCall:output:0random_zoom_9052*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_random_zoom_layer_call_and_return_conditional_losses_9051Е
IdentityIdentity,random_zoom/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ┤┤╝
NoOpNoOp$^random_flip/StatefulPartitionedCall(^random_rotation/StatefulPartitionedCall$^random_zoom/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         ┤┤: : : 2J
#random_flip/StatefulPartitionedCall#random_flip/StatefulPartitionedCall2R
'random_rotation/StatefulPartitionedCall'random_rotation/StatefulPartitionedCall2J
#random_zoom/StatefulPartitionedCall#random_zoom/StatefulPartitionedCall:d `
1
_output_shapes
:         ┤┤
+
_user_specified_namerandom_flip_input
П
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_7242

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Чй
В
D__inference_sequential_layer_call_and_return_conditional_losses_8471

inputse
Wmodule_wrapper_sequential_random_flip_stateful_uniform_full_int_rngreadandskip_resource:	`
Rmodule_wrapper_sequential_random_rotation_stateful_uniform_rngreadandskip_resource:	\
Nmodule_wrapper_sequential_random_zoom_stateful_uniform_rngreadandskip_resource:	?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource: 6
(conv2d_1_biasadd_readvariableop_resource: A
'conv2d_2_conv2d_readvariableop_resource: @6
(conv2d_2_biasadd_readvariableop_resource:@9
$dense_matmul_readvariableop_resource:АЄА4
%dense_biasadd_readvariableop_resource:	А9
&outputs_matmul_readvariableop_resource:	А5
'outputs_biasadd_readvariableop_resource:
identityИвconv2d/BiasAdd/ReadVariableOpвconv2d/Conv2D/ReadVariableOpвconv2d_1/BiasAdd/ReadVariableOpвconv2d_1/Conv2D/ReadVariableOpвconv2d_2/BiasAdd/ReadVariableOpвconv2d_2/Conv2D/ReadVariableOpвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpвNmodule_wrapper/sequential/random_flip/stateful_uniform_full_int/RngReadAndSkipвImodule_wrapper/sequential/random_rotation/stateful_uniform/RngReadAndSkipвEmodule_wrapper/sequential/random_zoom/stateful_uniform/RngReadAndSkipвoutputs/BiasAdd/ReadVariableOpвoutputs/MatMul/ReadVariableOpП
Emodule_wrapper/sequential/random_flip/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:П
Emodule_wrapper/sequential/random_flip/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: Н
Dmodule_wrapper/sequential/random_flip/stateful_uniform_full_int/ProdProdNmodule_wrapper/sequential/random_flip/stateful_uniform_full_int/shape:output:0Nmodule_wrapper/sequential/random_flip/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: И
Fmodule_wrapper/sequential/random_flip/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :═
Fmodule_wrapper/sequential/random_flip/stateful_uniform_full_int/Cast_1CastMmodule_wrapper/sequential/random_flip/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: Є
Nmodule_wrapper/sequential/random_flip/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkipWmodule_wrapper_sequential_random_flip_stateful_uniform_full_int_rngreadandskip_resourceOmodule_wrapper/sequential/random_flip/stateful_uniform_full_int/Cast/x:output:0Jmodule_wrapper/sequential/random_flip/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:Э
Smodule_wrapper/sequential/random_flip/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Я
Umodule_wrapper/sequential/random_flip/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Я
Umodule_wrapper/sequential/random_flip/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
Mmodule_wrapper/sequential/random_flip/stateful_uniform_full_int/strided_sliceStridedSliceVmodule_wrapper/sequential/random_flip/stateful_uniform_full_int/RngReadAndSkip:value:0\module_wrapper/sequential/random_flip/stateful_uniform_full_int/strided_slice/stack:output:0^module_wrapper/sequential/random_flip/stateful_uniform_full_int/strided_slice/stack_1:output:0^module_wrapper/sequential/random_flip/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask█
Gmodule_wrapper/sequential/random_flip/stateful_uniform_full_int/BitcastBitcastVmodule_wrapper/sequential/random_flip/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0Я
Umodule_wrapper/sequential/random_flip/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:б
Wmodule_wrapper/sequential/random_flip/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:б
Wmodule_wrapper/sequential/random_flip/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
Omodule_wrapper/sequential/random_flip/stateful_uniform_full_int/strided_slice_1StridedSliceVmodule_wrapper/sequential/random_flip/stateful_uniform_full_int/RngReadAndSkip:value:0^module_wrapper/sequential/random_flip/stateful_uniform_full_int/strided_slice_1/stack:output:0`module_wrapper/sequential/random_flip/stateful_uniform_full_int/strided_slice_1/stack_1:output:0`module_wrapper/sequential/random_flip/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:▀
Imodule_wrapper/sequential/random_flip/stateful_uniform_full_int/Bitcast_1BitcastXmodule_wrapper/sequential/random_flip/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0Е
Cmodule_wrapper/sequential/random_flip/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :╧
?module_wrapper/sequential/random_flip/stateful_uniform_full_intStatelessRandomUniformFullIntV2Nmodule_wrapper/sequential/random_flip/stateful_uniform_full_int/shape:output:0Rmodule_wrapper/sequential/random_flip/stateful_uniform_full_int/Bitcast_1:output:0Pmodule_wrapper/sequential/random_flip/stateful_uniform_full_int/Bitcast:output:0Lmodule_wrapper/sequential/random_flip/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	z
0module_wrapper/sequential/random_flip/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R ъ
+module_wrapper/sequential/random_flip/stackPackHmodule_wrapper/sequential/random_flip/stateful_uniform_full_int:output:09module_wrapper/sequential/random_flip/zeros_like:output:0*
N*
T0	*
_output_shapes

:К
9module_wrapper/sequential/random_flip/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        М
;module_wrapper/sequential/random_flip/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       М
;module_wrapper/sequential/random_flip/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╡
3module_wrapper/sequential/random_flip/strided_sliceStridedSlice4module_wrapper/sequential/random_flip/stack:output:0Bmodule_wrapper/sequential/random_flip/strided_slice/stack:output:0Dmodule_wrapper/sequential/random_flip/strided_slice/stack_1:output:0Dmodule_wrapper/sequential/random_flip/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask─
Ymodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:         ┤┤ь
Lmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/ShapeShapebmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
::э╧д
Zmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ж
\module_wrapper/sequential/random_flip/stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ж
\module_wrapper/sequential/random_flip/stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┤
Tmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/strided_sliceStridedSliceUmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/Shape:output:0cmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/strided_slice/stack:output:0emodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/strided_slice/stack_1:output:0emodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask·
emodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/shapePack]module_wrapper/sequential/random_flip/stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:и
cmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    и
cmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  А?№
|module_wrapper/sequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter<module_wrapper/sequential/random_flip/strided_slice:output:0* 
_output_shapes
::╛
|module_wrapper/sequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :┐
xmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2nmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/shape:output:0Вmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Жmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Еmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:         ч
cmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/subSublmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/max:output:0lmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: Е
cmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/mulMulБmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0gmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:         э
_module_wrapper/sequential/random_flip/stateless_random_flip_left_right/stateless_random_uniformAddV2gmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/mul:z:0lmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:         Ш
Vmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ш
Vmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Ш
Vmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :М
Tmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/Reshape/shapePack]module_wrapper/sequential/random_flip/stateless_random_flip_left_right/strided_slice:output:0_module_wrapper/sequential/random_flip/stateless_random_flip_left_right/Reshape/shape/1:output:0_module_wrapper/sequential/random_flip/stateless_random_flip_left_right/Reshape/shape/2:output:0_module_wrapper/sequential/random_flip/stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:╫
Nmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/ReshapeReshapecmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/stateless_random_uniform:z:0]module_wrapper/sequential/random_flip/stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:         ш
Lmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/RoundRoundWmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:         Я
Umodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:▌
Pmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/ReverseV2	ReverseV2bmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/control_dependency:output:0^module_wrapper/sequential/random_flip/stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:         ┤┤║
Jmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/mulMulPmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/Round:y:0Ymodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:         ┤┤С
Lmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?┤
Jmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/subSubUmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/sub/x:output:0Pmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:         ├
Lmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/mul_1MulNmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/sub:z:0bmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:         ┤┤▒
Jmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/addAddV2Nmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/mul:z:0Pmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:         ┤┤╗
/module_wrapper/sequential/random_rotation/ShapeShapeNmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/add:z:0*
T0*
_output_shapes
::э╧З
=module_wrapper/sequential/random_rotation/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Й
?module_wrapper/sequential/random_rotation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Й
?module_wrapper/sequential/random_rotation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
7module_wrapper/sequential/random_rotation/strided_sliceStridedSlice8module_wrapper/sequential/random_rotation/Shape:output:0Fmodule_wrapper/sequential/random_rotation/strided_slice/stack:output:0Hmodule_wrapper/sequential/random_rotation/strided_slice/stack_1:output:0Hmodule_wrapper/sequential/random_rotation/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskТ
?module_wrapper/sequential/random_rotation/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
¤        Ф
Amodule_wrapper/sequential/random_rotation/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
■        Л
Amodule_wrapper/sequential/random_rotation/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
9module_wrapper/sequential/random_rotation/strided_slice_1StridedSlice8module_wrapper/sequential/random_rotation/Shape:output:0Hmodule_wrapper/sequential/random_rotation/strided_slice_1/stack:output:0Jmodule_wrapper/sequential/random_rotation/strided_slice_1/stack_1:output:0Jmodule_wrapper/sequential/random_rotation/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskк
.module_wrapper/sequential/random_rotation/CastCastBmodule_wrapper/sequential/random_rotation/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: Т
?module_wrapper/sequential/random_rotation/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        Ф
Amodule_wrapper/sequential/random_rotation/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         Л
Amodule_wrapper/sequential/random_rotation/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
9module_wrapper/sequential/random_rotation/strided_slice_2StridedSlice8module_wrapper/sequential/random_rotation/Shape:output:0Hmodule_wrapper/sequential/random_rotation/strided_slice_2/stack:output:0Jmodule_wrapper/sequential/random_rotation/strided_slice_2/stack_1:output:0Jmodule_wrapper/sequential/random_rotation/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskм
0module_wrapper/sequential/random_rotation/Cast_1CastBmodule_wrapper/sequential/random_rotation/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: ╕
@module_wrapper/sequential/random_rotation/stateful_uniform/shapePack@module_wrapper/sequential/random_rotation/strided_slice:output:0*
N*
T0*
_output_shapes
:Г
>module_wrapper/sequential/random_rotation/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|┘ ┐Г
>module_wrapper/sequential/random_rotation/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|┘ ?К
@module_wrapper/sequential/random_rotation/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ■
?module_wrapper/sequential/random_rotation/stateful_uniform/ProdProdImodule_wrapper/sequential/random_rotation/stateful_uniform/shape:output:0Imodule_wrapper/sequential/random_rotation/stateful_uniform/Const:output:0*
T0*
_output_shapes
: Г
Amodule_wrapper/sequential/random_rotation/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :├
Amodule_wrapper/sequential/random_rotation/stateful_uniform/Cast_1CastHmodule_wrapper/sequential/random_rotation/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ▐
Imodule_wrapper/sequential/random_rotation/stateful_uniform/RngReadAndSkipRngReadAndSkipRmodule_wrapper_sequential_random_rotation_stateful_uniform_rngreadandskip_resourceJmodule_wrapper/sequential/random_rotation/stateful_uniform/Cast/x:output:0Emodule_wrapper/sequential/random_rotation/stateful_uniform/Cast_1:y:0*
_output_shapes
:Ш
Nmodule_wrapper/sequential/random_rotation/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ъ
Pmodule_wrapper/sequential/random_rotation/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ъ
Pmodule_wrapper/sequential/random_rotation/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:■
Hmodule_wrapper/sequential/random_rotation/stateful_uniform/strided_sliceStridedSliceQmodule_wrapper/sequential/random_rotation/stateful_uniform/RngReadAndSkip:value:0Wmodule_wrapper/sequential/random_rotation/stateful_uniform/strided_slice/stack:output:0Ymodule_wrapper/sequential/random_rotation/stateful_uniform/strided_slice/stack_1:output:0Ymodule_wrapper/sequential/random_rotation/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask╤
Bmodule_wrapper/sequential/random_rotation/stateful_uniform/BitcastBitcastQmodule_wrapper/sequential/random_rotation/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0Ъ
Pmodule_wrapper/sequential/random_rotation/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Ь
Rmodule_wrapper/sequential/random_rotation/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ь
Rmodule_wrapper/sequential/random_rotation/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
Jmodule_wrapper/sequential/random_rotation/stateful_uniform/strided_slice_1StridedSliceQmodule_wrapper/sequential/random_rotation/stateful_uniform/RngReadAndSkip:value:0Ymodule_wrapper/sequential/random_rotation/stateful_uniform/strided_slice_1/stack:output:0[module_wrapper/sequential/random_rotation/stateful_uniform/strided_slice_1/stack_1:output:0[module_wrapper/sequential/random_rotation/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:╒
Dmodule_wrapper/sequential/random_rotation/stateful_uniform/Bitcast_1BitcastSmodule_wrapper/sequential/random_rotation/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0Щ
Wmodule_wrapper/sequential/random_rotation/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :▌
Smodule_wrapper/sequential/random_rotation/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Imodule_wrapper/sequential/random_rotation/stateful_uniform/shape:output:0Mmodule_wrapper/sequential/random_rotation/stateful_uniform/Bitcast_1:output:0Kmodule_wrapper/sequential/random_rotation/stateful_uniform/Bitcast:output:0`module_wrapper/sequential/random_rotation/stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:         °
>module_wrapper/sequential/random_rotation/stateful_uniform/subSubGmodule_wrapper/sequential/random_rotation/stateful_uniform/max:output:0Gmodule_wrapper/sequential/random_rotation/stateful_uniform/min:output:0*
T0*
_output_shapes
: Х
>module_wrapper/sequential/random_rotation/stateful_uniform/mulMul\module_wrapper/sequential/random_rotation/stateful_uniform/StatelessRandomUniformV2:output:0Bmodule_wrapper/sequential/random_rotation/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:         ■
:module_wrapper/sequential/random_rotation/stateful_uniformAddV2Bmodule_wrapper/sequential/random_rotation/stateful_uniform/mul:z:0Gmodule_wrapper/sequential/random_rotation/stateful_uniform/min:output:0*
T0*#
_output_shapes
:         Д
?module_wrapper/sequential/random_rotation/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?х
=module_wrapper/sequential/random_rotation/rotation_matrix/subSub4module_wrapper/sequential/random_rotation/Cast_1:y:0Hmodule_wrapper/sequential/random_rotation/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: ▓
=module_wrapper/sequential/random_rotation/rotation_matrix/CosCos>module_wrapper/sequential/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:         Ж
Amodule_wrapper/sequential/random_rotation/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?щ
?module_wrapper/sequential/random_rotation/rotation_matrix/sub_1Sub4module_wrapper/sequential/random_rotation/Cast_1:y:0Jmodule_wrapper/sequential/random_rotation/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: ·
=module_wrapper/sequential/random_rotation/rotation_matrix/mulMulAmodule_wrapper/sequential/random_rotation/rotation_matrix/Cos:y:0Cmodule_wrapper/sequential/random_rotation/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:         ▓
=module_wrapper/sequential/random_rotation/rotation_matrix/SinSin>module_wrapper/sequential/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:         Ж
Amodule_wrapper/sequential/random_rotation/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?ч
?module_wrapper/sequential/random_rotation/rotation_matrix/sub_2Sub2module_wrapper/sequential/random_rotation/Cast:y:0Jmodule_wrapper/sequential/random_rotation/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: №
?module_wrapper/sequential/random_rotation/rotation_matrix/mul_1MulAmodule_wrapper/sequential/random_rotation/rotation_matrix/Sin:y:0Cmodule_wrapper/sequential/random_rotation/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:         №
?module_wrapper/sequential/random_rotation/rotation_matrix/sub_3SubAmodule_wrapper/sequential/random_rotation/rotation_matrix/mul:z:0Cmodule_wrapper/sequential/random_rotation/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:         №
?module_wrapper/sequential/random_rotation/rotation_matrix/sub_4SubAmodule_wrapper/sequential/random_rotation/rotation_matrix/sub:z:0Cmodule_wrapper/sequential/random_rotation/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:         И
Cmodule_wrapper/sequential/random_rotation/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @Н
Amodule_wrapper/sequential/random_rotation/rotation_matrix/truedivRealDivCmodule_wrapper/sequential/random_rotation/rotation_matrix/sub_4:z:0Lmodule_wrapper/sequential/random_rotation/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:         Ж
Amodule_wrapper/sequential/random_rotation/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?ч
?module_wrapper/sequential/random_rotation/rotation_matrix/sub_5Sub2module_wrapper/sequential/random_rotation/Cast:y:0Jmodule_wrapper/sequential/random_rotation/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: ┤
?module_wrapper/sequential/random_rotation/rotation_matrix/Sin_1Sin>module_wrapper/sequential/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:         Ж
Amodule_wrapper/sequential/random_rotation/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?щ
?module_wrapper/sequential/random_rotation/rotation_matrix/sub_6Sub4module_wrapper/sequential/random_rotation/Cast_1:y:0Jmodule_wrapper/sequential/random_rotation/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: ■
?module_wrapper/sequential/random_rotation/rotation_matrix/mul_2MulCmodule_wrapper/sequential/random_rotation/rotation_matrix/Sin_1:y:0Cmodule_wrapper/sequential/random_rotation/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:         ┤
?module_wrapper/sequential/random_rotation/rotation_matrix/Cos_1Cos>module_wrapper/sequential/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:         Ж
Amodule_wrapper/sequential/random_rotation/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?ч
?module_wrapper/sequential/random_rotation/rotation_matrix/sub_7Sub2module_wrapper/sequential/random_rotation/Cast:y:0Jmodule_wrapper/sequential/random_rotation/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: ■
?module_wrapper/sequential/random_rotation/rotation_matrix/mul_3MulCmodule_wrapper/sequential/random_rotation/rotation_matrix/Cos_1:y:0Cmodule_wrapper/sequential/random_rotation/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:         ■
=module_wrapper/sequential/random_rotation/rotation_matrix/addAddV2Cmodule_wrapper/sequential/random_rotation/rotation_matrix/mul_2:z:0Cmodule_wrapper/sequential/random_rotation/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:         №
?module_wrapper/sequential/random_rotation/rotation_matrix/sub_8SubCmodule_wrapper/sequential/random_rotation/rotation_matrix/sub_5:z:0Amodule_wrapper/sequential/random_rotation/rotation_matrix/add:z:0*
T0*#
_output_shapes
:         К
Emodule_wrapper/sequential/random_rotation/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @С
Cmodule_wrapper/sequential/random_rotation/rotation_matrix/truediv_1RealDivCmodule_wrapper/sequential/random_rotation/rotation_matrix/sub_8:z:0Nmodule_wrapper/sequential/random_rotation/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:         ╗
?module_wrapper/sequential/random_rotation/rotation_matrix/ShapeShape>module_wrapper/sequential/random_rotation/stateful_uniform:z:0*
T0*
_output_shapes
::э╧Ч
Mmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Щ
Omodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Щ
Omodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:є
Gmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_sliceStridedSliceHmodule_wrapper/sequential/random_rotation/rotation_matrix/Shape:output:0Vmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice/stack:output:0Xmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice/stack_1:output:0Xmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask┤
?module_wrapper/sequential/random_rotation/rotation_matrix/Cos_2Cos>module_wrapper/sequential/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:         а
Omodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        в
Qmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        в
Qmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ж
Imodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_1StridedSliceCmodule_wrapper/sequential/random_rotation/rotation_matrix/Cos_2:y:0Xmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_1/stack:output:0Zmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_1/stack_1:output:0Zmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask┤
?module_wrapper/sequential/random_rotation/rotation_matrix/Sin_2Sin>module_wrapper/sequential/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:         а
Omodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        в
Qmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        в
Qmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ж
Imodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_2StridedSliceCmodule_wrapper/sequential/random_rotation/rotation_matrix/Sin_2:y:0Xmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_2/stack:output:0Zmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_2/stack_1:output:0Zmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask╩
=module_wrapper/sequential/random_rotation/rotation_matrix/NegNegRmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:         а
Omodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        в
Qmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        в
Qmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      и
Imodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_3StridedSliceEmodule_wrapper/sequential/random_rotation/rotation_matrix/truediv:z:0Xmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_3/stack:output:0Zmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_3/stack_1:output:0Zmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask┤
?module_wrapper/sequential/random_rotation/rotation_matrix/Sin_3Sin>module_wrapper/sequential/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:         а
Omodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        в
Qmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        в
Qmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ж
Imodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_4StridedSliceCmodule_wrapper/sequential/random_rotation/rotation_matrix/Sin_3:y:0Xmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_4/stack:output:0Zmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_4/stack_1:output:0Zmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask┤
?module_wrapper/sequential/random_rotation/rotation_matrix/Cos_3Cos>module_wrapper/sequential/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:         а
Omodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        в
Qmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        в
Qmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ж
Imodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_5StridedSliceCmodule_wrapper/sequential/random_rotation/rotation_matrix/Cos_3:y:0Xmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_5/stack:output:0Zmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_5/stack_1:output:0Zmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_maskа
Omodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        в
Qmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        в
Qmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      к
Imodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_6StridedSliceGmodule_wrapper/sequential/random_rotation/rotation_matrix/truediv_1:z:0Xmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_6/stack:output:0Zmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_6/stack_1:output:0Zmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_maskК
Hmodule_wrapper/sequential/random_rotation/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :б
Fmodule_wrapper/sequential/random_rotation/rotation_matrix/zeros/packedPackPmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice:output:0Qmodule_wrapper/sequential/random_rotation/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:К
Emodule_wrapper/sequential/random_rotation/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ъ
?module_wrapper/sequential/random_rotation/rotation_matrix/zerosFillOmodule_wrapper/sequential/random_rotation/rotation_matrix/zeros/packed:output:0Nmodule_wrapper/sequential/random_rotation/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:         З
Emodule_wrapper/sequential/random_rotation/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :И
@module_wrapper/sequential/random_rotation/rotation_matrix/concatConcatV2Rmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_1:output:0Amodule_wrapper/sequential/random_rotation/rotation_matrix/Neg:y:0Rmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_3:output:0Rmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_4:output:0Rmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_5:output:0Rmodule_wrapper/sequential/random_rotation/rotation_matrix/strided_slice_6:output:0Hmodule_wrapper/sequential/random_rotation/rotation_matrix/zeros:output:0Nmodule_wrapper/sequential/random_rotation/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:         ┼
9module_wrapper/sequential/random_rotation/transform/ShapeShapeNmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/add:z:0*
T0*
_output_shapes
::э╧С
Gmodule_wrapper/sequential/random_rotation/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:У
Imodule_wrapper/sequential/random_rotation/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:У
Imodule_wrapper/sequential/random_rotation/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┴
Amodule_wrapper/sequential/random_rotation/transform/strided_sliceStridedSliceBmodule_wrapper/sequential/random_rotation/transform/Shape:output:0Pmodule_wrapper/sequential/random_rotation/transform/strided_slice/stack:output:0Rmodule_wrapper/sequential/random_rotation/transform/strided_slice/stack_1:output:0Rmodule_wrapper/sequential/random_rotation/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:Г
>module_wrapper/sequential/random_rotation/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    С
Nmodule_wrapper/sequential/random_rotation/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3Nmodule_wrapper/sequential/random_flip/stateless_random_flip_left_right/add:z:0Imodule_wrapper/sequential/random_rotation/rotation_matrix/concat:output:0Jmodule_wrapper/sequential/random_rotation/transform/strided_slice:output:0Gmodule_wrapper/sequential/random_rotation/transform/fill_value:output:0*1
_output_shapes
:         ┤┤*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR╠
+module_wrapper/sequential/random_zoom/ShapeShapecmodule_wrapper/sequential/random_rotation/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
::э╧Г
9module_wrapper/sequential/random_zoom/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Е
;module_wrapper/sequential/random_zoom/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Е
;module_wrapper/sequential/random_zoom/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:П
3module_wrapper/sequential/random_zoom/strided_sliceStridedSlice4module_wrapper/sequential/random_zoom/Shape:output:0Bmodule_wrapper/sequential/random_zoom/strided_slice/stack:output:0Dmodule_wrapper/sequential/random_zoom/strided_slice/stack_1:output:0Dmodule_wrapper/sequential/random_zoom/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskО
;module_wrapper/sequential/random_zoom/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
¤        Р
=module_wrapper/sequential/random_zoom/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
■        З
=module_wrapper/sequential/random_zoom/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
5module_wrapper/sequential/random_zoom/strided_slice_1StridedSlice4module_wrapper/sequential/random_zoom/Shape:output:0Dmodule_wrapper/sequential/random_zoom/strided_slice_1/stack:output:0Fmodule_wrapper/sequential/random_zoom/strided_slice_1/stack_1:output:0Fmodule_wrapper/sequential/random_zoom/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskв
*module_wrapper/sequential/random_zoom/CastCast>module_wrapper/sequential/random_zoom/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: О
;module_wrapper/sequential/random_zoom/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        Р
=module_wrapper/sequential/random_zoom/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         З
=module_wrapper/sequential/random_zoom/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
5module_wrapper/sequential/random_zoom/strided_slice_2StridedSlice4module_wrapper/sequential/random_zoom/Shape:output:0Dmodule_wrapper/sequential/random_zoom/strided_slice_2/stack:output:0Fmodule_wrapper/sequential/random_zoom/strided_slice_2/stack_1:output:0Fmodule_wrapper/sequential/random_zoom/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskд
,module_wrapper/sequential/random_zoom/Cast_1Cast>module_wrapper/sequential/random_zoom/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: А
>module_wrapper/sequential/random_zoom/stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :∙
<module_wrapper/sequential/random_zoom/stateful_uniform/shapePack<module_wrapper/sequential/random_zoom/strided_slice:output:0Gmodule_wrapper/sequential/random_zoom/stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:
:module_wrapper/sequential/random_zoom/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *fff?
:module_wrapper/sequential/random_zoom/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *═╠М?Ж
<module_wrapper/sequential/random_zoom/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: Є
;module_wrapper/sequential/random_zoom/stateful_uniform/ProdProdEmodule_wrapper/sequential/random_zoom/stateful_uniform/shape:output:0Emodule_wrapper/sequential/random_zoom/stateful_uniform/Const:output:0*
T0*
_output_shapes
: 
=module_wrapper/sequential/random_zoom/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :╗
=module_wrapper/sequential/random_zoom/stateful_uniform/Cast_1CastDmodule_wrapper/sequential/random_zoom/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ╬
Emodule_wrapper/sequential/random_zoom/stateful_uniform/RngReadAndSkipRngReadAndSkipNmodule_wrapper_sequential_random_zoom_stateful_uniform_rngreadandskip_resourceFmodule_wrapper/sequential/random_zoom/stateful_uniform/Cast/x:output:0Amodule_wrapper/sequential/random_zoom/stateful_uniform/Cast_1:y:0*
_output_shapes
:Ф
Jmodule_wrapper/sequential/random_zoom/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ц
Lmodule_wrapper/sequential/random_zoom/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ц
Lmodule_wrapper/sequential/random_zoom/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
Dmodule_wrapper/sequential/random_zoom/stateful_uniform/strided_sliceStridedSliceMmodule_wrapper/sequential/random_zoom/stateful_uniform/RngReadAndSkip:value:0Smodule_wrapper/sequential/random_zoom/stateful_uniform/strided_slice/stack:output:0Umodule_wrapper/sequential/random_zoom/stateful_uniform/strided_slice/stack_1:output:0Umodule_wrapper/sequential/random_zoom/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask╔
>module_wrapper/sequential/random_zoom/stateful_uniform/BitcastBitcastMmodule_wrapper/sequential/random_zoom/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0Ц
Lmodule_wrapper/sequential/random_zoom/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:Ш
Nmodule_wrapper/sequential/random_zoom/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ш
Nmodule_wrapper/sequential/random_zoom/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:р
Fmodule_wrapper/sequential/random_zoom/stateful_uniform/strided_slice_1StridedSliceMmodule_wrapper/sequential/random_zoom/stateful_uniform/RngReadAndSkip:value:0Umodule_wrapper/sequential/random_zoom/stateful_uniform/strided_slice_1/stack:output:0Wmodule_wrapper/sequential/random_zoom/stateful_uniform/strided_slice_1/stack_1:output:0Wmodule_wrapper/sequential/random_zoom/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:═
@module_wrapper/sequential/random_zoom/stateful_uniform/Bitcast_1BitcastOmodule_wrapper/sequential/random_zoom/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0Х
Smodule_wrapper/sequential/random_zoom/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :═
Omodule_wrapper/sequential/random_zoom/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Emodule_wrapper/sequential/random_zoom/stateful_uniform/shape:output:0Imodule_wrapper/sequential/random_zoom/stateful_uniform/Bitcast_1:output:0Gmodule_wrapper/sequential/random_zoom/stateful_uniform/Bitcast:output:0\module_wrapper/sequential/random_zoom/stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:         ь
:module_wrapper/sequential/random_zoom/stateful_uniform/subSubCmodule_wrapper/sequential/random_zoom/stateful_uniform/max:output:0Cmodule_wrapper/sequential/random_zoom/stateful_uniform/min:output:0*
T0*
_output_shapes
: Н
:module_wrapper/sequential/random_zoom/stateful_uniform/mulMulXmodule_wrapper/sequential/random_zoom/stateful_uniform/StatelessRandomUniformV2:output:0>module_wrapper/sequential/random_zoom/stateful_uniform/sub:z:0*
T0*'
_output_shapes
:         Ў
6module_wrapper/sequential/random_zoom/stateful_uniformAddV2>module_wrapper/sequential/random_zoom/stateful_uniform/mul:z:0Cmodule_wrapper/sequential/random_zoom/stateful_uniform/min:output:0*
T0*'
_output_shapes
:         s
1module_wrapper/sequential/random_zoom/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :з
,module_wrapper/sequential/random_zoom/concatConcatV2:module_wrapper/sequential/random_zoom/stateful_uniform:z:0:module_wrapper/sequential/random_zoom/stateful_uniform:z:0:module_wrapper/sequential/random_zoom/concat/axis:output:0*
N*
T0*'
_output_shapes
:         к
7module_wrapper/sequential/random_zoom/zoom_matrix/ShapeShape5module_wrapper/sequential/random_zoom/concat:output:0*
T0*
_output_shapes
::э╧П
Emodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: С
Gmodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:С
Gmodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╦
?module_wrapper/sequential/random_zoom/zoom_matrix/strided_sliceStridedSlice@module_wrapper/sequential/random_zoom/zoom_matrix/Shape:output:0Nmodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice/stack:output:0Pmodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice/stack_1:output:0Pmodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
7module_wrapper/sequential/random_zoom/zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╤
5module_wrapper/sequential/random_zoom/zoom_matrix/subSub0module_wrapper/sequential/random_zoom/Cast_1:y:0@module_wrapper/sequential/random_zoom/zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: А
;module_wrapper/sequential/random_zoom/zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @ц
9module_wrapper/sequential/random_zoom/zoom_matrix/truedivRealDiv9module_wrapper/sequential/random_zoom/zoom_matrix/sub:z:0Dmodule_wrapper/sequential/random_zoom/zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: Ь
Gmodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            Ю
Imodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           Ю
Imodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Р
Amodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice_1StridedSlice5module_wrapper/sequential/random_zoom/concat:output:0Pmodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice_1/stack:output:0Rmodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice_1/stack_1:output:0Rmodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask~
9module_wrapper/sequential/random_zoom/zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?А
7module_wrapper/sequential/random_zoom/zoom_matrix/sub_1SubBmodule_wrapper/sequential/random_zoom/zoom_matrix/sub_1/x:output:0Jmodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:         ъ
5module_wrapper/sequential/random_zoom/zoom_matrix/mulMul=module_wrapper/sequential/random_zoom/zoom_matrix/truediv:z:0;module_wrapper/sequential/random_zoom/zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:         ~
9module_wrapper/sequential/random_zoom/zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╙
7module_wrapper/sequential/random_zoom/zoom_matrix/sub_2Sub.module_wrapper/sequential/random_zoom/Cast:y:0Bmodule_wrapper/sequential/random_zoom/zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: В
=module_wrapper/sequential/random_zoom/zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @ь
;module_wrapper/sequential/random_zoom/zoom_matrix/truediv_1RealDiv;module_wrapper/sequential/random_zoom/zoom_matrix/sub_2:z:0Fmodule_wrapper/sequential/random_zoom/zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: Ь
Gmodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           Ю
Imodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           Ю
Imodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Р
Amodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice_2StridedSlice5module_wrapper/sequential/random_zoom/concat:output:0Pmodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice_2/stack:output:0Rmodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice_2/stack_1:output:0Rmodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask~
9module_wrapper/sequential/random_zoom/zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?А
7module_wrapper/sequential/random_zoom/zoom_matrix/sub_3SubBmodule_wrapper/sequential/random_zoom/zoom_matrix/sub_3/x:output:0Jmodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:         ю
7module_wrapper/sequential/random_zoom/zoom_matrix/mul_1Mul?module_wrapper/sequential/random_zoom/zoom_matrix/truediv_1:z:0;module_wrapper/sequential/random_zoom/zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:         Ь
Gmodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            Ю
Imodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           Ю
Imodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Р
Amodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice_3StridedSlice5module_wrapper/sequential/random_zoom/concat:output:0Pmodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice_3/stack:output:0Rmodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice_3/stack_1:output:0Rmodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskВ
@module_wrapper/sequential/random_zoom/zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Й
>module_wrapper/sequential/random_zoom/zoom_matrix/zeros/packedPackHmodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice:output:0Imodule_wrapper/sequential/random_zoom/zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:В
=module_wrapper/sequential/random_zoom/zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    В
7module_wrapper/sequential/random_zoom/zoom_matrix/zerosFillGmodule_wrapper/sequential/random_zoom/zoom_matrix/zeros/packed:output:0Fmodule_wrapper/sequential/random_zoom/zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:         Д
Bmodule_wrapper/sequential/random_zoom/zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Н
@module_wrapper/sequential/random_zoom/zoom_matrix/zeros_1/packedPackHmodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice:output:0Kmodule_wrapper/sequential/random_zoom/zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Д
?module_wrapper/sequential/random_zoom/zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    И
9module_wrapper/sequential/random_zoom/zoom_matrix/zeros_1FillImodule_wrapper/sequential/random_zoom/zoom_matrix/zeros_1/packed:output:0Hmodule_wrapper/sequential/random_zoom/zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:         Ь
Gmodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           Ю
Imodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           Ю
Imodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Р
Amodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice_4StridedSlice5module_wrapper/sequential/random_zoom/concat:output:0Pmodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice_4/stack:output:0Rmodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice_4/stack_1:output:0Rmodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskД
Bmodule_wrapper/sequential/random_zoom/zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Н
@module_wrapper/sequential/random_zoom/zoom_matrix/zeros_2/packedPackHmodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice:output:0Kmodule_wrapper/sequential/random_zoom/zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:Д
?module_wrapper/sequential/random_zoom/zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    И
9module_wrapper/sequential/random_zoom/zoom_matrix/zeros_2FillImodule_wrapper/sequential/random_zoom/zoom_matrix/zeros_2/packed:output:0Hmodule_wrapper/sequential/random_zoom/zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:         
=module_wrapper/sequential/random_zoom/zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :б
8module_wrapper/sequential/random_zoom/zoom_matrix/concatConcatV2Jmodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice_3:output:0@module_wrapper/sequential/random_zoom/zoom_matrix/zeros:output:09module_wrapper/sequential/random_zoom/zoom_matrix/mul:z:0Bmodule_wrapper/sequential/random_zoom/zoom_matrix/zeros_1:output:0Jmodule_wrapper/sequential/random_zoom/zoom_matrix/strided_slice_4:output:0;module_wrapper/sequential/random_zoom/zoom_matrix/mul_1:z:0Bmodule_wrapper/sequential/random_zoom/zoom_matrix/zeros_2:output:0Fmodule_wrapper/sequential/random_zoom/zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:         ╓
5module_wrapper/sequential/random_zoom/transform/ShapeShapecmodule_wrapper/sequential/random_rotation/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
::э╧Н
Cmodule_wrapper/sequential/random_zoom/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:П
Emodule_wrapper/sequential/random_zoom/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:П
Emodule_wrapper/sequential/random_zoom/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
=module_wrapper/sequential/random_zoom/transform/strided_sliceStridedSlice>module_wrapper/sequential/random_zoom/transform/Shape:output:0Lmodule_wrapper/sequential/random_zoom/transform/strided_slice/stack:output:0Nmodule_wrapper/sequential/random_zoom/transform/strided_slice/stack_1:output:0Nmodule_wrapper/sequential/random_zoom/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:
:module_wrapper/sequential/random_zoom/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    Т
Jmodule_wrapper/sequential/random_zoom/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3cmodule_wrapper/sequential/random_rotation/transform/ImageProjectiveTransformV3:transformed_images:0Amodule_wrapper/sequential/random_zoom/zoom_matrix/concat:output:0Fmodule_wrapper/sequential/random_zoom/transform/strided_slice:output:0Cmodule_wrapper/sequential/random_zoom/transform/fill_value:output:0*1
_output_shapes
:         ┤┤*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEARh
#module_wrapper_1/rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;j
%module_wrapper_1/rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Є
 module_wrapper_1/rescaling_1/mulMul_module_wrapper/sequential/random_zoom/transform/ImageProjectiveTransformV3:transformed_images:0,module_wrapper_1/rescaling_1/Cast/x:output:0*
T0*1
_output_shapes
:         ┤┤╗
 module_wrapper_1/rescaling_1/addAddV2$module_wrapper_1/rescaling_1/mul:z:0.module_wrapper_1/rescaling_1/Cast_1/x:output:0*
T0*1
_output_shapes
:         ┤┤К
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╟
conv2d/Conv2DConv2D$module_wrapper_1/rescaling_1/add:z:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ┤┤*
paddingSAME*
strides
А
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ф
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ┤┤h
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:         ┤┤и
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:         ZZ*
ksize
*
paddingVALID*
strides
О
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0├
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ZZ *
paddingSAME*
strides
Д
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ш
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ZZ j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         ZZ м
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:         -- *
ksize
*
paddingVALID*
strides
О
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0┼
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         --@*
paddingSAME*
strides
Д
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ш
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         --@j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:         --@м
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?Ц
dropout/dropout/MulMul max_pooling2d_2/MaxPool:output:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:         @s
dropout/dropout/ShapeShape max_pooling2d_2/MaxPool:output:0*
T0*
_output_shapes
::э╧д
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>╞
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @\
dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ╗
dropout/dropout/SelectV2SelectV2 dropout/dropout/GreaterEqual:z:0dropout/dropout/Mul:z:0 dropout/dropout/Const_1:output:0*
T0*/
_output_shapes
:         @^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"     y  Й
flatten/ReshapeReshape!dropout/dropout/SelectV2:output:0flatten/Const:output:0*
T0*)
_output_shapes
:         АЄГ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*!
_output_shapes
:АЄА*
dtype0И
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:         АЕ
outputs/MatMul/ReadVariableOpReadVariableOp&outputs_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Л
outputs/MatMulMatMuldense/Relu:activations:0%outputs/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         В
outputs/BiasAdd/ReadVariableOpReadVariableOp'outputs_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
outputs/BiasAddBiasAddoutputs/MatMul:product:0&outputs/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         g
IdentityIdentityoutputs/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         ю
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOpO^module_wrapper/sequential/random_flip/stateful_uniform_full_int/RngReadAndSkipJ^module_wrapper/sequential/random_rotation/stateful_uniform/RngReadAndSkipF^module_wrapper/sequential/random_zoom/stateful_uniform/RngReadAndSkip^outputs/BiasAdd/ReadVariableOp^outputs/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         ┤┤: : : : : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2а
Nmodule_wrapper/sequential/random_flip/stateful_uniform_full_int/RngReadAndSkipNmodule_wrapper/sequential/random_flip/stateful_uniform_full_int/RngReadAndSkip2Ц
Imodule_wrapper/sequential/random_rotation/stateful_uniform/RngReadAndSkipImodule_wrapper/sequential/random_rotation/stateful_uniform/RngReadAndSkip2О
Emodule_wrapper/sequential/random_zoom/stateful_uniform/RngReadAndSkipEmodule_wrapper/sequential/random_zoom/stateful_uniform/RngReadAndSkip2@
outputs/BiasAdd/ReadVariableOpoutputs/BiasAdd/ReadVariableOp2>
outputs/MatMul/ReadVariableOpoutputs/MatMul/ReadVariableOp:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameinputs
э

Щ
)__inference_sequential_layer_call_fn_7421
module_wrapper_input!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@
	unknown_5:АЄА
	unknown_6:	А
	unknown_7:	А
	unknown_8:
identityИвStatefulPartitionedCall╧
StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_7398o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:         ┤┤: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
1
_output_shapes
:         ┤┤
.
_user_specified_namemodule_wrapper_input
я
╜
D__inference_sequential_layer_call_and_return_conditional_losses_9091

inputs
random_flip_9081:	"
random_rotation_9084:	
random_zoom_9087:	
identityИв#random_flip/StatefulPartitionedCallв'random_rotation/StatefulPartitionedCallв#random_zoom/StatefulPartitionedCallщ
#random_flip/StatefulPartitionedCallStatefulPartitionedCallinputsrandom_flip_9081*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_random_flip_layer_call_and_return_conditional_losses_8823Ы
'random_rotation/StatefulPartitionedCallStatefulPartitionedCall,random_flip/StatefulPartitionedCall:output:0random_rotation_9084*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_random_rotation_layer_call_and_return_conditional_losses_8945У
#random_zoom/StatefulPartitionedCallStatefulPartitionedCall0random_rotation/StatefulPartitionedCall:output:0random_zoom_9087*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_random_zoom_layer_call_and_return_conditional_losses_9051Е
IdentityIdentity,random_zoom/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ┤┤╝
NoOpNoOp$^random_flip/StatefulPartitionedCall(^random_rotation/StatefulPartitionedCall$^random_zoom/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         ┤┤: : : 2J
#random_flip/StatefulPartitionedCall#random_flip/StatefulPartitionedCall2R
'random_rotation/StatefulPartitionedCall'random_rotation/StatefulPartitionedCall2J
#random_zoom/StatefulPartitionedCall#random_zoom/StatefulPartitionedCall:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameinputs
┼
]
A__inference_flatten_layer_call_and_return_conditional_losses_7362

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"     y  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:         АЄZ
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:         АЄ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
ў▓
ь%
__inference__traced_save_10236
file_prefix>
$read_disablecopyonread_conv2d_kernel:2
$read_1_disablecopyonread_conv2d_bias:B
(read_2_disablecopyonread_conv2d_1_kernel: 4
&read_3_disablecopyonread_conv2d_1_bias: B
(read_4_disablecopyonread_conv2d_2_kernel: @4
&read_5_disablecopyonread_conv2d_2_bias:@:
%read_6_disablecopyonread_dense_kernel:АЄА2
#read_7_disablecopyonread_dense_bias:	А:
'read_8_disablecopyonread_outputs_kernel:	А3
%read_9_disablecopyonread_outputs_bias:-
#read_10_disablecopyonread_adam_iter:	 /
%read_11_disablecopyonread_adam_beta_1: /
%read_12_disablecopyonread_adam_beta_2: .
$read_13_disablecopyonread_adam_decay: 6
,read_14_disablecopyonread_adam_learning_rate: +
!read_15_disablecopyonread_total_1: +
!read_16_disablecopyonread_count_1: )
read_17_disablecopyonread_total: )
read_18_disablecopyonread_count: <
.read_19_disablecopyonread_random_flip_statevar:	@
2read_20_disablecopyonread_random_rotation_statevar:	<
.read_21_disablecopyonread_random_zoom_statevar:	H
.read_22_disablecopyonread_adam_conv2d_kernel_m::
,read_23_disablecopyonread_adam_conv2d_bias_m:J
0read_24_disablecopyonread_adam_conv2d_1_kernel_m: <
.read_25_disablecopyonread_adam_conv2d_1_bias_m: J
0read_26_disablecopyonread_adam_conv2d_2_kernel_m: @<
.read_27_disablecopyonread_adam_conv2d_2_bias_m:@B
-read_28_disablecopyonread_adam_dense_kernel_m:АЄА:
+read_29_disablecopyonread_adam_dense_bias_m:	АB
/read_30_disablecopyonread_adam_outputs_kernel_m:	А;
-read_31_disablecopyonread_adam_outputs_bias_m:H
.read_32_disablecopyonread_adam_conv2d_kernel_v::
,read_33_disablecopyonread_adam_conv2d_bias_v:J
0read_34_disablecopyonread_adam_conv2d_1_kernel_v: <
.read_35_disablecopyonread_adam_conv2d_1_bias_v: J
0read_36_disablecopyonread_adam_conv2d_2_kernel_v: @<
.read_37_disablecopyonread_adam_conv2d_2_bias_v:@B
-read_38_disablecopyonread_adam_dense_kernel_v:АЄА:
+read_39_disablecopyonread_adam_dense_bias_v:	АB
/read_40_disablecopyonread_adam_outputs_kernel_v:	А;
-read_41_disablecopyonread_adam_outputs_bias_v:
savev2_const
identity_85ИвMergeV2CheckpointsвRead/DisableCopyOnReadвRead/ReadVariableOpвRead_1/DisableCopyOnReadвRead_1/ReadVariableOpвRead_10/DisableCopyOnReadвRead_10/ReadVariableOpвRead_11/DisableCopyOnReadвRead_11/ReadVariableOpвRead_12/DisableCopyOnReadвRead_12/ReadVariableOpвRead_13/DisableCopyOnReadвRead_13/ReadVariableOpвRead_14/DisableCopyOnReadвRead_14/ReadVariableOpвRead_15/DisableCopyOnReadвRead_15/ReadVariableOpвRead_16/DisableCopyOnReadвRead_16/ReadVariableOpвRead_17/DisableCopyOnReadвRead_17/ReadVariableOpвRead_18/DisableCopyOnReadвRead_18/ReadVariableOpвRead_19/DisableCopyOnReadвRead_19/ReadVariableOpвRead_2/DisableCopyOnReadвRead_2/ReadVariableOpвRead_20/DisableCopyOnReadвRead_20/ReadVariableOpвRead_21/DisableCopyOnReadвRead_21/ReadVariableOpвRead_22/DisableCopyOnReadвRead_22/ReadVariableOpвRead_23/DisableCopyOnReadвRead_23/ReadVariableOpвRead_24/DisableCopyOnReadвRead_24/ReadVariableOpвRead_25/DisableCopyOnReadвRead_25/ReadVariableOpвRead_26/DisableCopyOnReadвRead_26/ReadVariableOpвRead_27/DisableCopyOnReadвRead_27/ReadVariableOpвRead_28/DisableCopyOnReadвRead_28/ReadVariableOpвRead_29/DisableCopyOnReadвRead_29/ReadVariableOpвRead_3/DisableCopyOnReadвRead_3/ReadVariableOpвRead_30/DisableCopyOnReadвRead_30/ReadVariableOpвRead_31/DisableCopyOnReadвRead_31/ReadVariableOpвRead_32/DisableCopyOnReadвRead_32/ReadVariableOpвRead_33/DisableCopyOnReadвRead_33/ReadVariableOpвRead_34/DisableCopyOnReadвRead_34/ReadVariableOpвRead_35/DisableCopyOnReadвRead_35/ReadVariableOpвRead_36/DisableCopyOnReadвRead_36/ReadVariableOpвRead_37/DisableCopyOnReadвRead_37/ReadVariableOpвRead_38/DisableCopyOnReadвRead_38/ReadVariableOpвRead_39/DisableCopyOnReadвRead_39/ReadVariableOpвRead_4/DisableCopyOnReadвRead_4/ReadVariableOpвRead_40/DisableCopyOnReadвRead_40/ReadVariableOpвRead_41/DisableCopyOnReadвRead_41/ReadVariableOpвRead_5/DisableCopyOnReadвRead_5/ReadVariableOpвRead_6/DisableCopyOnReadвRead_6/ReadVariableOpвRead_7/DisableCopyOnReadвRead_7/ReadVariableOpвRead_8/DisableCopyOnReadвRead_8/ReadVariableOpвRead_9/DisableCopyOnReadвRead_9/ReadVariableOpw
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
_temp/partБ
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
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: v
Read/DisableCopyOnReadDisableCopyOnRead$read_disablecopyonread_conv2d_kernel"/device:CPU:0*
_output_shapes
 и
Read/ReadVariableOpReadVariableOp$read_disablecopyonread_conv2d_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
:x
Read_1/DisableCopyOnReadDisableCopyOnRead$read_1_disablecopyonread_conv2d_bias"/device:CPU:0*
_output_shapes
 а
Read_1/ReadVariableOpReadVariableOp$read_1_disablecopyonread_conv2d_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_2/DisableCopyOnReadDisableCopyOnRead(read_2_disablecopyonread_conv2d_1_kernel"/device:CPU:0*
_output_shapes
 ░
Read_2/ReadVariableOpReadVariableOp(read_2_disablecopyonread_conv2d_1_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0u

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: k

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*&
_output_shapes
: z
Read_3/DisableCopyOnReadDisableCopyOnRead&read_3_disablecopyonread_conv2d_1_bias"/device:CPU:0*
_output_shapes
 в
Read_3/ReadVariableOpReadVariableOp&read_3_disablecopyonread_conv2d_1_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
: |
Read_4/DisableCopyOnReadDisableCopyOnRead(read_4_disablecopyonread_conv2d_2_kernel"/device:CPU:0*
_output_shapes
 ░
Read_4/ReadVariableOpReadVariableOp(read_4_disablecopyonread_conv2d_2_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0u

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @k

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*&
_output_shapes
: @z
Read_5/DisableCopyOnReadDisableCopyOnRead&read_5_disablecopyonread_conv2d_2_bias"/device:CPU:0*
_output_shapes
 в
Read_5/ReadVariableOpReadVariableOp&read_5_disablecopyonread_conv2d_2_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:@y
Read_6/DisableCopyOnReadDisableCopyOnRead%read_6_disablecopyonread_dense_kernel"/device:CPU:0*
_output_shapes
 и
Read_6/ReadVariableOpReadVariableOp%read_6_disablecopyonread_dense_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*!
_output_shapes
:АЄА*
dtype0q
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*!
_output_shapes
:АЄАh
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*!
_output_shapes
:АЄАw
Read_7/DisableCopyOnReadDisableCopyOnRead#read_7_disablecopyonread_dense_bias"/device:CPU:0*
_output_shapes
 а
Read_7/ReadVariableOpReadVariableOp#read_7_disablecopyonread_dense_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0k
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes	
:А{
Read_8/DisableCopyOnReadDisableCopyOnRead'read_8_disablecopyonread_outputs_kernel"/device:CPU:0*
_output_shapes
 и
Read_8/ReadVariableOpReadVariableOp'read_8_disablecopyonread_outputs_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А*
dtype0o
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	Аf
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:	Аy
Read_9/DisableCopyOnReadDisableCopyOnRead%read_9_disablecopyonread_outputs_bias"/device:CPU:0*
_output_shapes
 б
Read_9/ReadVariableOpReadVariableOp%read_9_disablecopyonread_outputs_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
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
:x
Read_10/DisableCopyOnReadDisableCopyOnRead#read_10_disablecopyonread_adam_iter"/device:CPU:0*
_output_shapes
 Э
Read_10/ReadVariableOpReadVariableOp#read_10_disablecopyonread_adam_iter^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_11/DisableCopyOnReadDisableCopyOnRead%read_11_disablecopyonread_adam_beta_1"/device:CPU:0*
_output_shapes
 Я
Read_11/ReadVariableOpReadVariableOp%read_11_disablecopyonread_adam_beta_1^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
: z
Read_12/DisableCopyOnReadDisableCopyOnRead%read_12_disablecopyonread_adam_beta_2"/device:CPU:0*
_output_shapes
 Я
Read_12/ReadVariableOpReadVariableOp%read_12_disablecopyonread_adam_beta_2^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
: y
Read_13/DisableCopyOnReadDisableCopyOnRead$read_13_disablecopyonread_adam_decay"/device:CPU:0*
_output_shapes
 Ю
Read_13/ReadVariableOpReadVariableOp$read_13_disablecopyonread_adam_decay^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
: Б
Read_14/DisableCopyOnReadDisableCopyOnRead,read_14_disablecopyonread_adam_learning_rate"/device:CPU:0*
_output_shapes
 ж
Read_14/ReadVariableOpReadVariableOp,read_14_disablecopyonread_adam_learning_rate^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_15/DisableCopyOnReadDisableCopyOnRead!read_15_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 Ы
Read_15/ReadVariableOpReadVariableOp!read_15_disablecopyonread_total_1^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_16/DisableCopyOnReadDisableCopyOnRead!read_16_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 Ы
Read_16/ReadVariableOpReadVariableOp!read_16_disablecopyonread_count_1^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_17/DisableCopyOnReadDisableCopyOnReadread_17_disablecopyonread_total"/device:CPU:0*
_output_shapes
 Щ
Read_17/ReadVariableOpReadVariableOpread_17_disablecopyonread_total^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_18/DisableCopyOnReadDisableCopyOnReadread_18_disablecopyonread_count"/device:CPU:0*
_output_shapes
 Щ
Read_18/ReadVariableOpReadVariableOpread_18_disablecopyonread_count^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
: Г
Read_19/DisableCopyOnReadDisableCopyOnRead.read_19_disablecopyonread_random_flip_statevar"/device:CPU:0*
_output_shapes
 м
Read_19/ReadVariableOpReadVariableOp.read_19_disablecopyonread_random_flip_statevar^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0	k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
:a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0	*
_output_shapes
:З
Read_20/DisableCopyOnReadDisableCopyOnRead2read_20_disablecopyonread_random_rotation_statevar"/device:CPU:0*
_output_shapes
 ░
Read_20/ReadVariableOpReadVariableOp2read_20_disablecopyonread_random_rotation_statevar^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0	k
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
:a
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0	*
_output_shapes
:Г
Read_21/DisableCopyOnReadDisableCopyOnRead.read_21_disablecopyonread_random_zoom_statevar"/device:CPU:0*
_output_shapes
 м
Read_21/ReadVariableOpReadVariableOp.read_21_disablecopyonread_random_zoom_statevar^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0	k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
:a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0	*
_output_shapes
:Г
Read_22/DisableCopyOnReadDisableCopyOnRead.read_22_disablecopyonread_adam_conv2d_kernel_m"/device:CPU:0*
_output_shapes
 ╕
Read_22/ReadVariableOpReadVariableOp.read_22_disablecopyonread_adam_conv2d_kernel_m^Read_22/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*&
_output_shapes
:Б
Read_23/DisableCopyOnReadDisableCopyOnRead,read_23_disablecopyonread_adam_conv2d_bias_m"/device:CPU:0*
_output_shapes
 к
Read_23/ReadVariableOpReadVariableOp,read_23_disablecopyonread_adam_conv2d_bias_m^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:Е
Read_24/DisableCopyOnReadDisableCopyOnRead0read_24_disablecopyonread_adam_conv2d_1_kernel_m"/device:CPU:0*
_output_shapes
 ║
Read_24/ReadVariableOpReadVariableOp0read_24_disablecopyonread_adam_conv2d_1_kernel_m^Read_24/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*&
_output_shapes
: Г
Read_25/DisableCopyOnReadDisableCopyOnRead.read_25_disablecopyonread_adam_conv2d_1_bias_m"/device:CPU:0*
_output_shapes
 м
Read_25/ReadVariableOpReadVariableOp.read_25_disablecopyonread_adam_conv2d_1_bias_m^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
: Е
Read_26/DisableCopyOnReadDisableCopyOnRead0read_26_disablecopyonread_adam_conv2d_2_kernel_m"/device:CPU:0*
_output_shapes
 ║
Read_26/ReadVariableOpReadVariableOp0read_26_disablecopyonread_adam_conv2d_2_kernel_m^Read_26/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0w
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @m
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*&
_output_shapes
: @Г
Read_27/DisableCopyOnReadDisableCopyOnRead.read_27_disablecopyonread_adam_conv2d_2_bias_m"/device:CPU:0*
_output_shapes
 м
Read_27/ReadVariableOpReadVariableOp.read_27_disablecopyonread_adam_conv2d_2_bias_m^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:@В
Read_28/DisableCopyOnReadDisableCopyOnRead-read_28_disablecopyonread_adam_dense_kernel_m"/device:CPU:0*
_output_shapes
 ▓
Read_28/ReadVariableOpReadVariableOp-read_28_disablecopyonread_adam_dense_kernel_m^Read_28/DisableCopyOnRead"/device:CPU:0*!
_output_shapes
:АЄА*
dtype0r
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*!
_output_shapes
:АЄАh
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*!
_output_shapes
:АЄАА
Read_29/DisableCopyOnReadDisableCopyOnRead+read_29_disablecopyonread_adam_dense_bias_m"/device:CPU:0*
_output_shapes
 к
Read_29/ReadVariableOpReadVariableOp+read_29_disablecopyonread_adam_dense_bias_m^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes	
:АД
Read_30/DisableCopyOnReadDisableCopyOnRead/read_30_disablecopyonread_adam_outputs_kernel_m"/device:CPU:0*
_output_shapes
 ▓
Read_30/ReadVariableOpReadVariableOp/read_30_disablecopyonread_adam_outputs_kernel_m^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А*
dtype0p
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	Аf
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
:	АВ
Read_31/DisableCopyOnReadDisableCopyOnRead-read_31_disablecopyonread_adam_outputs_bias_m"/device:CPU:0*
_output_shapes
 л
Read_31/ReadVariableOpReadVariableOp-read_31_disablecopyonread_adam_outputs_bias_m^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
:Г
Read_32/DisableCopyOnReadDisableCopyOnRead.read_32_disablecopyonread_adam_conv2d_kernel_v"/device:CPU:0*
_output_shapes
 ╕
Read_32/ReadVariableOpReadVariableOp.read_32_disablecopyonread_adam_conv2d_kernel_v^Read_32/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*&
_output_shapes
:Б
Read_33/DisableCopyOnReadDisableCopyOnRead,read_33_disablecopyonread_adam_conv2d_bias_v"/device:CPU:0*
_output_shapes
 к
Read_33/ReadVariableOpReadVariableOp,read_33_disablecopyonread_adam_conv2d_bias_v^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
:Е
Read_34/DisableCopyOnReadDisableCopyOnRead0read_34_disablecopyonread_adam_conv2d_1_kernel_v"/device:CPU:0*
_output_shapes
 ║
Read_34/ReadVariableOpReadVariableOp0read_34_disablecopyonread_adam_conv2d_1_kernel_v^Read_34/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*&
_output_shapes
: Г
Read_35/DisableCopyOnReadDisableCopyOnRead.read_35_disablecopyonread_adam_conv2d_1_bias_v"/device:CPU:0*
_output_shapes
 м
Read_35/ReadVariableOpReadVariableOp.read_35_disablecopyonread_adam_conv2d_1_bias_v^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
: Е
Read_36/DisableCopyOnReadDisableCopyOnRead0read_36_disablecopyonread_adam_conv2d_2_kernel_v"/device:CPU:0*
_output_shapes
 ║
Read_36/ReadVariableOpReadVariableOp0read_36_disablecopyonread_adam_conv2d_2_kernel_v^Read_36/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0w
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @m
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*&
_output_shapes
: @Г
Read_37/DisableCopyOnReadDisableCopyOnRead.read_37_disablecopyonread_adam_conv2d_2_bias_v"/device:CPU:0*
_output_shapes
 м
Read_37/ReadVariableOpReadVariableOp.read_37_disablecopyonread_adam_conv2d_2_bias_v^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
:@В
Read_38/DisableCopyOnReadDisableCopyOnRead-read_38_disablecopyonread_adam_dense_kernel_v"/device:CPU:0*
_output_shapes
 ▓
Read_38/ReadVariableOpReadVariableOp-read_38_disablecopyonread_adam_dense_kernel_v^Read_38/DisableCopyOnRead"/device:CPU:0*!
_output_shapes
:АЄА*
dtype0r
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*!
_output_shapes
:АЄАh
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*!
_output_shapes
:АЄАА
Read_39/DisableCopyOnReadDisableCopyOnRead+read_39_disablecopyonread_adam_dense_bias_v"/device:CPU:0*
_output_shapes
 к
Read_39/ReadVariableOpReadVariableOp+read_39_disablecopyonread_adam_dense_bias_v^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes	
:АД
Read_40/DisableCopyOnReadDisableCopyOnRead/read_40_disablecopyonread_adam_outputs_kernel_v"/device:CPU:0*
_output_shapes
 ▓
Read_40/ReadVariableOpReadVariableOp/read_40_disablecopyonread_adam_outputs_kernel_v^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А*
dtype0p
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	Аf
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes
:	АВ
Read_41/DisableCopyOnReadDisableCopyOnRead-read_41_disablecopyonread_adam_outputs_bias_v"/device:CPU:0*
_output_shapes
 л
Read_41/ReadVariableOpReadVariableOp-read_41_disablecopyonread_adam_outputs_bias_v^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes
:¤
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*ж
valueЬBЩ+B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBZlayer-0/_module/layer-0/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer-0/_module/layer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer-0/_module/layer-2/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH├
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Щ	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *9
dtypes/
-2+				Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_84Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_85IdentityIdentity_84:output:0^NoOp*
T0*
_output_shapes
: ї
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_85Identity_85:output:0*k
_input_shapesZ
X: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
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
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:+

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
└ї
╙
D__inference_sequential_layer_call_and_return_conditional_losses_9616

inputsK
=random_flip_stateful_uniform_full_int_rngreadandskip_resource:	F
8random_rotation_stateful_uniform_rngreadandskip_resource:	B
4random_zoom_stateful_uniform_rngreadandskip_resource:	
identityИв4random_flip/stateful_uniform_full_int/RngReadAndSkipв/random_rotation/stateful_uniform/RngReadAndSkipв+random_zoom/stateful_uniform/RngReadAndSkipu
+random_flip/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:u
+random_flip/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: ┐
*random_flip/stateful_uniform_full_int/ProdProd4random_flip/stateful_uniform_full_int/shape:output:04random_flip/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: n
,random_flip/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :Щ
,random_flip/stateful_uniform_full_int/Cast_1Cast3random_flip/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: К
4random_flip/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip=random_flip_stateful_uniform_full_int_rngreadandskip_resource5random_flip/stateful_uniform_full_int/Cast/x:output:00random_flip/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:Г
9random_flip/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Е
;random_flip/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Е
;random_flip/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Х
3random_flip/stateful_uniform_full_int/strided_sliceStridedSlice<random_flip/stateful_uniform_full_int/RngReadAndSkip:value:0Brandom_flip/stateful_uniform_full_int/strided_slice/stack:output:0Drandom_flip/stateful_uniform_full_int/strided_slice/stack_1:output:0Drandom_flip/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskз
-random_flip/stateful_uniform_full_int/BitcastBitcast<random_flip/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0Е
;random_flip/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:З
=random_flip/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:З
=random_flip/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
5random_flip/stateful_uniform_full_int/strided_slice_1StridedSlice<random_flip/stateful_uniform_full_int/RngReadAndSkip:value:0Drandom_flip/stateful_uniform_full_int/strided_slice_1/stack:output:0Frandom_flip/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Frandom_flip/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:л
/random_flip/stateful_uniform_full_int/Bitcast_1Bitcast>random_flip/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0k
)random_flip/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :═
%random_flip/stateful_uniform_full_intStatelessRandomUniformFullIntV24random_flip/stateful_uniform_full_int/shape:output:08random_flip/stateful_uniform_full_int/Bitcast_1:output:06random_flip/stateful_uniform_full_int/Bitcast:output:02random_flip/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	`
random_flip/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R Ь
random_flip/stackPack.random_flip/stateful_uniform_full_int:output:0random_flip/zeros_like:output:0*
N*
T0	*
_output_shapes

:p
random_flip/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!random_flip/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!random_flip/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      │
random_flip/strided_sliceStridedSlicerandom_flip/stack:output:0(random_flip/strided_slice/stack:output:0*random_flip/strided_slice/stack_1:output:0*random_flip/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_maskк
?random_flip/stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:         ┤┤╕
2random_flip/stateless_random_flip_left_right/ShapeShapeHrandom_flip/stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
::э╧К
@random_flip/stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: М
Brandom_flip/stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:М
Brandom_flip/stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:▓
:random_flip/stateless_random_flip_left_right/strided_sliceStridedSlice;random_flip/stateless_random_flip_left_right/Shape:output:0Irandom_flip/stateless_random_flip_left_right/strided_slice/stack:output:0Krandom_flip/stateless_random_flip_left_right/strided_slice/stack_1:output:0Krandom_flip/stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╞
Krandom_flip/stateless_random_flip_left_right/stateless_random_uniform/shapePackCrandom_flip/stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:О
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    О
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╚
brandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter"random_flip/strided_slice:output:0* 
_output_shapes
::д
brandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :║
^random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Trandom_flip/stateless_random_flip_left_right/stateless_random_uniform/shape:output:0hrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0lrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0krandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:         Щ
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/subSubRrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/max:output:0Rrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ╢
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/mulMulgrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Mrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:         Я
Erandom_flip/stateless_random_flip_left_right/stateless_random_uniformAddV2Mrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/mul:z:0Rrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:         ~
<random_flip/stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
<random_flip/stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :~
<random_flip/stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :К
:random_flip/stateless_random_flip_left_right/Reshape/shapePackCrandom_flip/stateless_random_flip_left_right/strided_slice:output:0Erandom_flip/stateless_random_flip_left_right/Reshape/shape/1:output:0Erandom_flip/stateless_random_flip_left_right/Reshape/shape/2:output:0Erandom_flip/stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Й
4random_flip/stateless_random_flip_left_right/ReshapeReshapeIrandom_flip/stateless_random_flip_left_right/stateless_random_uniform:z:0Crandom_flip/stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:         ┤
2random_flip/stateless_random_flip_left_right/RoundRound=random_flip/stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:         Е
;random_flip/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:П
6random_flip/stateless_random_flip_left_right/ReverseV2	ReverseV2Hrandom_flip/stateless_random_flip_left_right/control_dependency:output:0Drandom_flip/stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:         ┤┤ь
0random_flip/stateless_random_flip_left_right/mulMul6random_flip/stateless_random_flip_left_right/Round:y:0?random_flip/stateless_random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:         ┤┤w
2random_flip/stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?ц
0random_flip/stateless_random_flip_left_right/subSub;random_flip/stateless_random_flip_left_right/sub/x:output:06random_flip/stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:         ї
2random_flip/stateless_random_flip_left_right/mul_1Mul4random_flip/stateless_random_flip_left_right/sub:z:0Hrandom_flip/stateless_random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:         ┤┤у
0random_flip/stateless_random_flip_left_right/addAddV24random_flip/stateless_random_flip_left_right/mul:z:06random_flip/stateless_random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:         ┤┤З
random_rotation/ShapeShape4random_flip/stateless_random_flip_left_right/add:z:0*
T0*
_output_shapes
::э╧m
#random_rotation/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%random_rotation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%random_rotation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
random_rotation/strided_sliceStridedSlicerandom_rotation/Shape:output:0,random_rotation/strided_slice/stack:output:0.random_rotation/strided_slice/stack_1:output:0.random_rotation/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
%random_rotation/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
¤        z
'random_rotation/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
■        q
'random_rotation/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
random_rotation/strided_slice_1StridedSlicerandom_rotation/Shape:output:0.random_rotation/strided_slice_1/stack:output:00random_rotation/strided_slice_1/stack_1:output:00random_rotation/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
random_rotation/CastCast(random_rotation/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: x
%random_rotation/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        z
'random_rotation/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         q
'random_rotation/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
random_rotation/strided_slice_2StridedSlicerandom_rotation/Shape:output:0.random_rotation/strided_slice_2/stack:output:00random_rotation/strided_slice_2/stack_1:output:00random_rotation/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
random_rotation/Cast_1Cast(random_rotation/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: Д
&random_rotation/stateful_uniform/shapePack&random_rotation/strided_slice:output:0*
N*
T0*
_output_shapes
:i
$random_rotation/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|┘ ┐i
$random_rotation/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|┘ ?p
&random_rotation/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ░
%random_rotation/stateful_uniform/ProdProd/random_rotation/stateful_uniform/shape:output:0/random_rotation/stateful_uniform/Const:output:0*
T0*
_output_shapes
: i
'random_rotation/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :П
'random_rotation/stateful_uniform/Cast_1Cast.random_rotation/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: Ў
/random_rotation/stateful_uniform/RngReadAndSkipRngReadAndSkip8random_rotation_stateful_uniform_rngreadandskip_resource0random_rotation/stateful_uniform/Cast/x:output:0+random_rotation/stateful_uniform/Cast_1:y:0*
_output_shapes
:~
4random_rotation/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: А
6random_rotation/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:А
6random_rotation/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:№
.random_rotation/stateful_uniform/strided_sliceStridedSlice7random_rotation/stateful_uniform/RngReadAndSkip:value:0=random_rotation/stateful_uniform/strided_slice/stack:output:0?random_rotation/stateful_uniform/strided_slice/stack_1:output:0?random_rotation/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskЭ
(random_rotation/stateful_uniform/BitcastBitcast7random_rotation/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0А
6random_rotation/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:В
8random_rotation/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:В
8random_rotation/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
0random_rotation/stateful_uniform/strided_slice_1StridedSlice7random_rotation/stateful_uniform/RngReadAndSkip:value:0?random_rotation/stateful_uniform/strided_slice_1/stack:output:0Arandom_rotation/stateful_uniform/strided_slice_1/stack_1:output:0Arandom_rotation/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:б
*random_rotation/stateful_uniform/Bitcast_1Bitcast9random_rotation/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0
=random_rotation/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :█
9random_rotation/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2/random_rotation/stateful_uniform/shape:output:03random_rotation/stateful_uniform/Bitcast_1:output:01random_rotation/stateful_uniform/Bitcast:output:0Frandom_rotation/stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:         к
$random_rotation/stateful_uniform/subSub-random_rotation/stateful_uniform/max:output:0-random_rotation/stateful_uniform/min:output:0*
T0*
_output_shapes
: ╟
$random_rotation/stateful_uniform/mulMulBrandom_rotation/stateful_uniform/StatelessRandomUniformV2:output:0(random_rotation/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:         ░
 random_rotation/stateful_uniformAddV2(random_rotation/stateful_uniform/mul:z:0-random_rotation/stateful_uniform/min:output:0*
T0*#
_output_shapes
:         j
%random_rotation/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ч
#random_rotation/rotation_matrix/subSubrandom_rotation/Cast_1:y:0.random_rotation/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: ~
#random_rotation/rotation_matrix/CosCos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:         l
'random_rotation/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ы
%random_rotation/rotation_matrix/sub_1Subrandom_rotation/Cast_1:y:00random_rotation/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: м
#random_rotation/rotation_matrix/mulMul'random_rotation/rotation_matrix/Cos:y:0)random_rotation/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:         ~
#random_rotation/rotation_matrix/SinSin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:         l
'random_rotation/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Щ
%random_rotation/rotation_matrix/sub_2Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: о
%random_rotation/rotation_matrix/mul_1Mul'random_rotation/rotation_matrix/Sin:y:0)random_rotation/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:         о
%random_rotation/rotation_matrix/sub_3Sub'random_rotation/rotation_matrix/mul:z:0)random_rotation/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:         о
%random_rotation/rotation_matrix/sub_4Sub'random_rotation/rotation_matrix/sub:z:0)random_rotation/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:         n
)random_rotation/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @┐
'random_rotation/rotation_matrix/truedivRealDiv)random_rotation/rotation_matrix/sub_4:z:02random_rotation/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:         l
'random_rotation/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Щ
%random_rotation/rotation_matrix/sub_5Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: А
%random_rotation/rotation_matrix/Sin_1Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:         l
'random_rotation/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ы
%random_rotation/rotation_matrix/sub_6Subrandom_rotation/Cast_1:y:00random_rotation/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: ░
%random_rotation/rotation_matrix/mul_2Mul)random_rotation/rotation_matrix/Sin_1:y:0)random_rotation/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:         А
%random_rotation/rotation_matrix/Cos_1Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:         l
'random_rotation/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Щ
%random_rotation/rotation_matrix/sub_7Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: ░
%random_rotation/rotation_matrix/mul_3Mul)random_rotation/rotation_matrix/Cos_1:y:0)random_rotation/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:         ░
#random_rotation/rotation_matrix/addAddV2)random_rotation/rotation_matrix/mul_2:z:0)random_rotation/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:         о
%random_rotation/rotation_matrix/sub_8Sub)random_rotation/rotation_matrix/sub_5:z:0'random_rotation/rotation_matrix/add:z:0*
T0*#
_output_shapes
:         p
+random_rotation/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @├
)random_rotation/rotation_matrix/truediv_1RealDiv)random_rotation/rotation_matrix/sub_8:z:04random_rotation/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:         З
%random_rotation/rotation_matrix/ShapeShape$random_rotation/stateful_uniform:z:0*
T0*
_output_shapes
::э╧}
3random_rotation/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5random_rotation/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5random_rotation/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ё
-random_rotation/rotation_matrix/strided_sliceStridedSlice.random_rotation/rotation_matrix/Shape:output:0<random_rotation/rotation_matrix/strided_slice/stack:output:0>random_rotation/rotation_matrix/strided_slice/stack_1:output:0>random_rotation/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskА
%random_rotation/rotation_matrix/Cos_2Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:         Ж
5random_rotation/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        И
7random_rotation/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        И
7random_rotation/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      д
/random_rotation/rotation_matrix/strided_slice_1StridedSlice)random_rotation/rotation_matrix/Cos_2:y:0>random_rotation/rotation_matrix/strided_slice_1/stack:output:0@random_rotation/rotation_matrix/strided_slice_1/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_maskА
%random_rotation/rotation_matrix/Sin_2Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:         Ж
5random_rotation/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        И
7random_rotation/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        И
7random_rotation/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      д
/random_rotation/rotation_matrix/strided_slice_2StridedSlice)random_rotation/rotation_matrix/Sin_2:y:0>random_rotation/rotation_matrix/strided_slice_2/stack:output:0@random_rotation/rotation_matrix/strided_slice_2/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_maskЦ
#random_rotation/rotation_matrix/NegNeg8random_rotation/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:         Ж
5random_rotation/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        И
7random_rotation/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        И
7random_rotation/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ж
/random_rotation/rotation_matrix/strided_slice_3StridedSlice+random_rotation/rotation_matrix/truediv:z:0>random_rotation/rotation_matrix/strided_slice_3/stack:output:0@random_rotation/rotation_matrix/strided_slice_3/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_maskА
%random_rotation/rotation_matrix/Sin_3Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:         Ж
5random_rotation/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        И
7random_rotation/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        И
7random_rotation/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      д
/random_rotation/rotation_matrix/strided_slice_4StridedSlice)random_rotation/rotation_matrix/Sin_3:y:0>random_rotation/rotation_matrix/strided_slice_4/stack:output:0@random_rotation/rotation_matrix/strided_slice_4/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_maskА
%random_rotation/rotation_matrix/Cos_3Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:         Ж
5random_rotation/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        И
7random_rotation/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        И
7random_rotation/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      д
/random_rotation/rotation_matrix/strided_slice_5StridedSlice)random_rotation/rotation_matrix/Cos_3:y:0>random_rotation/rotation_matrix/strided_slice_5/stack:output:0@random_rotation/rotation_matrix/strided_slice_5/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_maskЖ
5random_rotation/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        И
7random_rotation/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        И
7random_rotation/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      и
/random_rotation/rotation_matrix/strided_slice_6StridedSlice-random_rotation/rotation_matrix/truediv_1:z:0>random_rotation/rotation_matrix/strided_slice_6/stack:output:0@random_rotation/rotation_matrix/strided_slice_6/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_maskp
.random_rotation/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :╙
,random_rotation/rotation_matrix/zeros/packedPack6random_rotation/rotation_matrix/strided_slice:output:07random_rotation/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:p
+random_rotation/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ╠
%random_rotation/rotation_matrix/zerosFill5random_rotation/rotation_matrix/zeros/packed:output:04random_rotation/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:         m
+random_rotation/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ю
&random_rotation/rotation_matrix/concatConcatV28random_rotation/rotation_matrix/strided_slice_1:output:0'random_rotation/rotation_matrix/Neg:y:08random_rotation/rotation_matrix/strided_slice_3:output:08random_rotation/rotation_matrix/strided_slice_4:output:08random_rotation/rotation_matrix/strided_slice_5:output:08random_rotation/rotation_matrix/strided_slice_6:output:0.random_rotation/rotation_matrix/zeros:output:04random_rotation/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:         С
random_rotation/transform/ShapeShape4random_flip/stateless_random_flip_left_right/add:z:0*
T0*
_output_shapes
::э╧w
-random_rotation/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:y
/random_rotation/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/random_rotation/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┐
'random_rotation/transform/strided_sliceStridedSlice(random_rotation/transform/Shape:output:06random_rotation/transform/strided_slice/stack:output:08random_rotation/transform/strided_slice/stack_1:output:08random_rotation/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:i
$random_rotation/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    П
4random_rotation/transform/ImageProjectiveTransformV3ImageProjectiveTransformV34random_flip/stateless_random_flip_left_right/add:z:0/random_rotation/rotation_matrix/concat:output:00random_rotation/transform/strided_slice:output:0-random_rotation/transform/fill_value:output:0*1
_output_shapes
:         ┤┤*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEARШ
random_zoom/ShapeShapeIrandom_rotation/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
::э╧i
random_zoom/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!random_zoom/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!random_zoom/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
random_zoom/strided_sliceStridedSlicerandom_zoom/Shape:output:0(random_zoom/strided_slice/stack:output:0*random_zoom/strided_slice/stack_1:output:0*random_zoom/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
!random_zoom/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
¤        v
#random_zoom/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
■        m
#random_zoom/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Х
random_zoom/strided_slice_1StridedSlicerandom_zoom/Shape:output:0*random_zoom/strided_slice_1/stack:output:0,random_zoom/strided_slice_1/stack_1:output:0,random_zoom/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
random_zoom/CastCast$random_zoom/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: t
!random_zoom/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        v
#random_zoom/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         m
#random_zoom/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Х
random_zoom/strided_slice_2StridedSlicerandom_zoom/Shape:output:0*random_zoom/strided_slice_2/stack:output:0,random_zoom/strided_slice_2/stack_1:output:0,random_zoom/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
random_zoom/Cast_1Cast$random_zoom/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: f
$random_zoom/stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :л
"random_zoom/stateful_uniform/shapePack"random_zoom/strided_slice:output:0-random_zoom/stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:e
 random_zoom/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *fff?e
 random_zoom/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *═╠М?l
"random_zoom/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: д
!random_zoom/stateful_uniform/ProdProd+random_zoom/stateful_uniform/shape:output:0+random_zoom/stateful_uniform/Const:output:0*
T0*
_output_shapes
: e
#random_zoom/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :З
#random_zoom/stateful_uniform/Cast_1Cast*random_zoom/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ц
+random_zoom/stateful_uniform/RngReadAndSkipRngReadAndSkip4random_zoom_stateful_uniform_rngreadandskip_resource,random_zoom/stateful_uniform/Cast/x:output:0'random_zoom/stateful_uniform/Cast_1:y:0*
_output_shapes
:z
0random_zoom/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2random_zoom/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2random_zoom/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ш
*random_zoom/stateful_uniform/strided_sliceStridedSlice3random_zoom/stateful_uniform/RngReadAndSkip:value:09random_zoom/stateful_uniform/strided_slice/stack:output:0;random_zoom/stateful_uniform/strided_slice/stack_1:output:0;random_zoom/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskХ
$random_zoom/stateful_uniform/BitcastBitcast3random_zoom/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0|
2random_zoom/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:~
4random_zoom/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4random_zoom/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:▐
,random_zoom/stateful_uniform/strided_slice_1StridedSlice3random_zoom/stateful_uniform/RngReadAndSkip:value:0;random_zoom/stateful_uniform/strided_slice_1/stack:output:0=random_zoom/stateful_uniform/strided_slice_1/stack_1:output:0=random_zoom/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:Щ
&random_zoom/stateful_uniform/Bitcast_1Bitcast5random_zoom/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0{
9random_zoom/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :╦
5random_zoom/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2+random_zoom/stateful_uniform/shape:output:0/random_zoom/stateful_uniform/Bitcast_1:output:0-random_zoom/stateful_uniform/Bitcast:output:0Brandom_zoom/stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:         Ю
 random_zoom/stateful_uniform/subSub)random_zoom/stateful_uniform/max:output:0)random_zoom/stateful_uniform/min:output:0*
T0*
_output_shapes
: ┐
 random_zoom/stateful_uniform/mulMul>random_zoom/stateful_uniform/StatelessRandomUniformV2:output:0$random_zoom/stateful_uniform/sub:z:0*
T0*'
_output_shapes
:         и
random_zoom/stateful_uniformAddV2$random_zoom/stateful_uniform/mul:z:0)random_zoom/stateful_uniform/min:output:0*
T0*'
_output_shapes
:         Y
random_zoom/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :┐
random_zoom/concatConcatV2 random_zoom/stateful_uniform:z:0 random_zoom/stateful_uniform:z:0 random_zoom/concat/axis:output:0*
N*
T0*'
_output_shapes
:         v
random_zoom/zoom_matrix/ShapeShaperandom_zoom/concat:output:0*
T0*
_output_shapes
::э╧u
+random_zoom/zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-random_zoom/zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-random_zoom/zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╔
%random_zoom/zoom_matrix/strided_sliceStridedSlice&random_zoom/zoom_matrix/Shape:output:04random_zoom/zoom_matrix/strided_slice/stack:output:06random_zoom/zoom_matrix/strided_slice/stack_1:output:06random_zoom/zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
random_zoom/zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Г
random_zoom/zoom_matrix/subSubrandom_zoom/Cast_1:y:0&random_zoom/zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: f
!random_zoom/zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @Ш
random_zoom/zoom_matrix/truedivRealDivrandom_zoom/zoom_matrix/sub:z:0*random_zoom/zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: В
-random_zoom/zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            Д
/random_zoom/zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           Д
/random_zoom/zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         О
'random_zoom/zoom_matrix/strided_slice_1StridedSlicerandom_zoom/concat:output:06random_zoom/zoom_matrix/strided_slice_1/stack:output:08random_zoom/zoom_matrix/strided_slice_1/stack_1:output:08random_zoom/zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskd
random_zoom/zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?▓
random_zoom/zoom_matrix/sub_1Sub(random_zoom/zoom_matrix/sub_1/x:output:00random_zoom/zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:         Ь
random_zoom/zoom_matrix/mulMul#random_zoom/zoom_matrix/truediv:z:0!random_zoom/zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:         d
random_zoom/zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Е
random_zoom/zoom_matrix/sub_2Subrandom_zoom/Cast:y:0(random_zoom/zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: h
#random_zoom/zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @Ю
!random_zoom/zoom_matrix/truediv_1RealDiv!random_zoom/zoom_matrix/sub_2:z:0,random_zoom/zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: В
-random_zoom/zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           Д
/random_zoom/zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           Д
/random_zoom/zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         О
'random_zoom/zoom_matrix/strided_slice_2StridedSlicerandom_zoom/concat:output:06random_zoom/zoom_matrix/strided_slice_2/stack:output:08random_zoom/zoom_matrix/strided_slice_2/stack_1:output:08random_zoom/zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskd
random_zoom/zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?▓
random_zoom/zoom_matrix/sub_3Sub(random_zoom/zoom_matrix/sub_3/x:output:00random_zoom/zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:         а
random_zoom/zoom_matrix/mul_1Mul%random_zoom/zoom_matrix/truediv_1:z:0!random_zoom/zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:         В
-random_zoom/zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            Д
/random_zoom/zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           Д
/random_zoom/zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         О
'random_zoom/zoom_matrix/strided_slice_3StridedSlicerandom_zoom/concat:output:06random_zoom/zoom_matrix/strided_slice_3/stack:output:08random_zoom/zoom_matrix/strided_slice_3/stack_1:output:08random_zoom/zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskh
&random_zoom/zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :╗
$random_zoom/zoom_matrix/zeros/packedPack.random_zoom/zoom_matrix/strided_slice:output:0/random_zoom/zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:h
#random_zoom/zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ┤
random_zoom/zoom_matrix/zerosFill-random_zoom/zoom_matrix/zeros/packed:output:0,random_zoom/zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:         j
(random_zoom/zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :┐
&random_zoom/zoom_matrix/zeros_1/packedPack.random_zoom/zoom_matrix/strided_slice:output:01random_zoom/zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:j
%random_zoom/zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ║
random_zoom/zoom_matrix/zeros_1Fill/random_zoom/zoom_matrix/zeros_1/packed:output:0.random_zoom/zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:         В
-random_zoom/zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           Д
/random_zoom/zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           Д
/random_zoom/zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         О
'random_zoom/zoom_matrix/strided_slice_4StridedSlicerandom_zoom/concat:output:06random_zoom/zoom_matrix/strided_slice_4/stack:output:08random_zoom/zoom_matrix/strided_slice_4/stack_1:output:08random_zoom/zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskj
(random_zoom/zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :┐
&random_zoom/zoom_matrix/zeros_2/packedPack.random_zoom/zoom_matrix/strided_slice:output:01random_zoom/zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:j
%random_zoom/zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ║
random_zoom/zoom_matrix/zeros_2Fill/random_zoom/zoom_matrix/zeros_2/packed:output:0.random_zoom/zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:         e
#random_zoom/zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :╖
random_zoom/zoom_matrix/concatConcatV20random_zoom/zoom_matrix/strided_slice_3:output:0&random_zoom/zoom_matrix/zeros:output:0random_zoom/zoom_matrix/mul:z:0(random_zoom/zoom_matrix/zeros_1:output:00random_zoom/zoom_matrix/strided_slice_4:output:0!random_zoom/zoom_matrix/mul_1:z:0(random_zoom/zoom_matrix/zeros_2:output:0,random_zoom/zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:         в
random_zoom/transform/ShapeShapeIrandom_rotation/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
::э╧s
)random_zoom/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+random_zoom/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+random_zoom/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
#random_zoom/transform/strided_sliceStridedSlice$random_zoom/transform/Shape:output:02random_zoom/transform/strided_slice/stack:output:04random_zoom/transform/strided_slice/stack_1:output:04random_zoom/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:e
 random_zoom/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    Р
0random_zoom/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3Irandom_rotation/transform/ImageProjectiveTransformV3:transformed_images:0'random_zoom/zoom_matrix/concat:output:0,random_zoom/transform/strided_slice:output:0)random_zoom/transform/fill_value:output:0*1
_output_shapes
:         ┤┤*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEARЮ
IdentityIdentityErandom_zoom/transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:         ┤┤▌
NoOpNoOp5^random_flip/stateful_uniform_full_int/RngReadAndSkip0^random_rotation/stateful_uniform/RngReadAndSkip,^random_zoom/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         ┤┤: : : 2l
4random_flip/stateful_uniform_full_int/RngReadAndSkip4random_flip/stateful_uniform_full_int/RngReadAndSkip2b
/random_rotation/stateful_uniform/RngReadAndSkip/random_rotation/stateful_uniform/RngReadAndSkip2Z
+random_zoom/stateful_uniform/RngReadAndSkip+random_zoom/stateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameinputs
Й
_
&__inference_dropout_layer_call_fn_9263

inputs
identityИвStatefulPartitionedCall─
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_7467w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
▌
f
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_9185

args_0
identityW
rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;Y
rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    w
rescaling_1/mulMulargs_0rescaling_1/Cast/x:output:0*
T0*1
_output_shapes
:         ┤┤И
rescaling_1/addAddV2rescaling_1/mul:z:0rescaling_1/Cast_1/x:output:0*
T0*1
_output_shapes
:         ┤┤e
IdentityIdentityrescaling_1/add:z:0*
T0*1
_output_shapes
:         ┤┤"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ┤┤:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameargs_0
╛А
┬
I__inference_random_rotation_layer_call_and_return_conditional_losses_8945

inputs6
(stateful_uniform_rngreadandskip_resource:	
identityИвstateful_uniform/RngReadAndSkipI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
¤        j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
■        a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: d
stateful_uniform/shapePackstrided_slice:output:0*
N*
T0*
_output_shapes
:Y
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|┘ ┐Y
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|┘ ?`
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: А
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: Y
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :o
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ╢
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:n
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:м
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask}
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0p
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:в
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:Б
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0o
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :Л
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:         z
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: Ч
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*#
_output_shapes
:         А
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*#
_output_shapes
:         Z
rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?g
rotation_matrix/subSub
Cast_1:y:0rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: ^
rotation_matrix/CosCosstateful_uniform:z:0*
T0*#
_output_shapes
:         \
rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?k
rotation_matrix/sub_1Sub
Cast_1:y:0 rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: |
rotation_matrix/mulMulrotation_matrix/Cos:y:0rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:         ^
rotation_matrix/SinSinstateful_uniform:z:0*
T0*#
_output_shapes
:         \
rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?i
rotation_matrix/sub_2SubCast:y:0 rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: ~
rotation_matrix/mul_1Mulrotation_matrix/Sin:y:0rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:         ~
rotation_matrix/sub_3Subrotation_matrix/mul:z:0rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:         ~
rotation_matrix/sub_4Subrotation_matrix/sub:z:0rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:         ^
rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @П
rotation_matrix/truedivRealDivrotation_matrix/sub_4:z:0"rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:         \
rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?i
rotation_matrix/sub_5SubCast:y:0 rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: `
rotation_matrix/Sin_1Sinstateful_uniform:z:0*
T0*#
_output_shapes
:         \
rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?k
rotation_matrix/sub_6Sub
Cast_1:y:0 rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: А
rotation_matrix/mul_2Mulrotation_matrix/Sin_1:y:0rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:         `
rotation_matrix/Cos_1Cosstateful_uniform:z:0*
T0*#
_output_shapes
:         \
rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?i
rotation_matrix/sub_7SubCast:y:0 rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: А
rotation_matrix/mul_3Mulrotation_matrix/Cos_1:y:0rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:         А
rotation_matrix/addAddV2rotation_matrix/mul_2:z:0rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:         ~
rotation_matrix/sub_8Subrotation_matrix/sub_5:z:0rotation_matrix/add:z:0*
T0*#
_output_shapes
:         `
rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @У
rotation_matrix/truediv_1RealDivrotation_matrix/sub_8:z:0$rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:         g
rotation_matrix/ShapeShapestateful_uniform:z:0*
T0*
_output_shapes
::э╧m
#rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
rotation_matrix/strided_sliceStridedSlicerotation_matrix/Shape:output:0,rotation_matrix/strided_slice/stack:output:0.rotation_matrix/strided_slice/stack_1:output:0.rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
rotation_matrix/Cos_2Cosstateful_uniform:z:0*
T0*#
_output_shapes
:         v
%rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
rotation_matrix/strided_slice_1StridedSlicerotation_matrix/Cos_2:y:0.rotation_matrix/strided_slice_1/stack:output:00rotation_matrix/strided_slice_1/stack_1:output:00rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Sin_2Sinstateful_uniform:z:0*
T0*#
_output_shapes
:         v
%rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
rotation_matrix/strided_slice_2StridedSlicerotation_matrix/Sin_2:y:0.rotation_matrix/strided_slice_2/stack:output:00rotation_matrix/strided_slice_2/stack_1:output:00rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_maskv
rotation_matrix/NegNeg(rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:         v
%rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
rotation_matrix/strided_slice_3StridedSlicerotation_matrix/truediv:z:0.rotation_matrix/strided_slice_3/stack:output:00rotation_matrix/strided_slice_3/stack_1:output:00rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Sin_3Sinstateful_uniform:z:0*
T0*#
_output_shapes
:         v
%rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
rotation_matrix/strided_slice_4StridedSlicerotation_matrix/Sin_3:y:0.rotation_matrix/strided_slice_4/stack:output:00rotation_matrix/strided_slice_4/stack_1:output:00rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Cos_3Cosstateful_uniform:z:0*
T0*#
_output_shapes
:         v
%rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
rotation_matrix/strided_slice_5StridedSlicerotation_matrix/Cos_3:y:0.rotation_matrix/strided_slice_5/stack:output:00rotation_matrix/strided_slice_5/stack_1:output:00rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_maskv
%rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╪
rotation_matrix/strided_slice_6StridedSlicerotation_matrix/truediv_1:z:0.rotation_matrix/strided_slice_6/stack:output:00rotation_matrix/strided_slice_6/stack_1:output:00rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :г
rotation_matrix/zeros/packedPack&rotation_matrix/strided_slice:output:0'rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:`
rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ь
rotation_matrix/zerosFill%rotation_matrix/zeros/packed:output:0$rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:         ]
rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :О
rotation_matrix/concatConcatV2(rotation_matrix/strided_slice_1:output:0rotation_matrix/Neg:y:0(rotation_matrix/strided_slice_3:output:0(rotation_matrix/strided_slice_4:output:0(rotation_matrix/strided_slice_5:output:0(rotation_matrix/strided_slice_6:output:0rotation_matrix/zeros:output:0$rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:         S
transform/ShapeShapeinputs*
T0*
_output_shapes
::э╧g
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:я
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:Y
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    б
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputsrotation_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*1
_output_shapes
:         ┤┤*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEARТ
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:         ┤┤h
NoOpNoOp ^stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ┤┤: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameinputs
Б
√
B__inference_conv2d_2_layer_call_and_return_conditional_losses_7342

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         --@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         --@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         --@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         --@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         -- : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         -- 
 
_user_specified_nameinputs
У
a
E__inference_random_zoom_layer_call_and_return_conditional_losses_9946

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:         ┤┤"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ┤┤:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameinputs
╧
J
.__inference_random_rotation_layer_call_fn_9706

inputs
identity╛
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_random_rotation_layer_call_and_return_conditional_losses_9067j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ┤┤"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ┤┤:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameinputs
▌
f
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_7516

args_0
identityW
rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;Y
rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    w
rescaling_1/mulMulargs_0rescaling_1/Cast/x:output:0*
T0*1
_output_shapes
:         ┤┤И
rescaling_1/addAddV2rescaling_1/mul:z:0rescaling_1/Cast_1/x:output:0*
T0*1
_output_shapes
:         ┤┤e
IdentityIdentityrescaling_1/add:z:0*
T0*1
_output_shapes
:         ┤┤"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ┤┤:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameargs_0
Ч
e
I__inference_random_rotation_layer_call_and_return_conditional_losses_9067

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:         ┤┤"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ┤┤:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameinputs
▌
f
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_9193

args_0
identityW
rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;Y
rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    w
rescaling_1/mulMulargs_0rescaling_1/Cast/x:output:0*
T0*1
_output_shapes
:         ┤┤И
rescaling_1/addAddV2rescaling_1/mul:z:0rescaling_1/Cast_1/x:output:0*
T0*1
_output_shapes
:         ┤┤e
IdentityIdentityrescaling_1/add:z:0*
T0*1
_output_shapes
:         ┤┤"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ┤┤:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameargs_0
Ф3
З
D__inference_sequential_layer_call_and_return_conditional_losses_7875

inputs!
module_wrapper_7836:	!
module_wrapper_7838:	!
module_wrapper_7840:	%
conv2d_7844:
conv2d_7846:'
conv2d_1_7850: 
conv2d_1_7852: '
conv2d_2_7856: @
conv2d_2_7858:@

dense_7864:АЄА

dense_7866:	А
outputs_7869:	А
outputs_7871:
identityИвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallв conv2d_2/StatefulPartitionedCallвdense/StatefulPartitionedCallвdropout/StatefulPartitionedCallв&module_wrapper/StatefulPartitionedCallвoutputs/StatefulPartitionedCallЮ
&module_wrapper/StatefulPartitionedCallStatefulPartitionedCallinputsmodule_wrapper_7836module_wrapper_7838module_wrapper_7840*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_module_wrapper_layer_call_and_return_conditional_losses_7797∙
 module_wrapper_1/PartitionedCallPartitionedCall/module_wrapper/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_7516П
conv2d/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_1/PartitionedCall:output:0conv2d_7844conv2d_7846*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_7306щ
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ZZ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_7242Т
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_7850conv2d_1_7852*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ZZ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_7324я
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         -- * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_7254Ф
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_7856conv2d_2_7858*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         --@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_7342я
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_7266ю
dropout/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_7467╪
flatten/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         АЄ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_7362∙
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dense_7864
dense_7866*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_7375Ж
outputs/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0outputs_7869outputs_7871*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_outputs_layer_call_and_return_conditional_losses_7391w
IdentityIdentity(outputs/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ║
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall'^module_wrapper/StatefulPartitionedCall ^outputs/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         ┤┤: : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2P
&module_wrapper/StatefulPartitionedCall&module_wrapper/StatefulPartitionedCall2B
outputs/StatefulPartitionedCalloutputs/StatefulPartitionedCall:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameinputs
н╡
┬
!__inference__traced_restore_10372
file_prefix8
assignvariableop_conv2d_kernel:,
assignvariableop_1_conv2d_bias:<
"assignvariableop_2_conv2d_1_kernel: .
 assignvariableop_3_conv2d_1_bias: <
"assignvariableop_4_conv2d_2_kernel: @.
 assignvariableop_5_conv2d_2_bias:@4
assignvariableop_6_dense_kernel:АЄА,
assignvariableop_7_dense_bias:	А4
!assignvariableop_8_outputs_kernel:	А-
assignvariableop_9_outputs_bias:'
assignvariableop_10_adam_iter:	 )
assignvariableop_11_adam_beta_1: )
assignvariableop_12_adam_beta_2: (
assignvariableop_13_adam_decay: 0
&assignvariableop_14_adam_learning_rate: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: #
assignvariableop_17_total: #
assignvariableop_18_count: 6
(assignvariableop_19_random_flip_statevar:	:
,assignvariableop_20_random_rotation_statevar:	6
(assignvariableop_21_random_zoom_statevar:	B
(assignvariableop_22_adam_conv2d_kernel_m:4
&assignvariableop_23_adam_conv2d_bias_m:D
*assignvariableop_24_adam_conv2d_1_kernel_m: 6
(assignvariableop_25_adam_conv2d_1_bias_m: D
*assignvariableop_26_adam_conv2d_2_kernel_m: @6
(assignvariableop_27_adam_conv2d_2_bias_m:@<
'assignvariableop_28_adam_dense_kernel_m:АЄА4
%assignvariableop_29_adam_dense_bias_m:	А<
)assignvariableop_30_adam_outputs_kernel_m:	А5
'assignvariableop_31_adam_outputs_bias_m:B
(assignvariableop_32_adam_conv2d_kernel_v:4
&assignvariableop_33_adam_conv2d_bias_v:D
*assignvariableop_34_adam_conv2d_1_kernel_v: 6
(assignvariableop_35_adam_conv2d_1_bias_v: D
*assignvariableop_36_adam_conv2d_2_kernel_v: @6
(assignvariableop_37_adam_conv2d_2_bias_v:@<
'assignvariableop_38_adam_dense_kernel_v:АЄА4
%assignvariableop_39_adam_dense_bias_v:	А<
)assignvariableop_40_adam_outputs_kernel_v:	А5
'assignvariableop_41_adam_outputs_bias_v:
identity_43ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9А
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*ж
valueЬBЩ+B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBZlayer-0/_module/layer-0/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer-0/_module/layer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer-0/_module/layer-2/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH╞
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B °
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*┬
_output_shapesп
м:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+				[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:▒
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:╡
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_2_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_2_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:╢
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_8AssignVariableOp!assignvariableop_8_outputs_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:╢
AssignVariableOp_9AssignVariableOpassignvariableop_9_outputs_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:╢
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0	*
_output_shapes
:┴
AssignVariableOp_19AssignVariableOp(assignvariableop_19_random_flip_statevarIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0	*
_output_shapes
:┼
AssignVariableOp_20AssignVariableOp,assignvariableop_20_random_rotation_statevarIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0	*
_output_shapes
:┴
AssignVariableOp_21AssignVariableOp(assignvariableop_21_random_zoom_statevarIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_conv2d_kernel_mIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_23AssignVariableOp&assignvariableop_23_adam_conv2d_bias_mIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_conv2d_1_kernel_mIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_conv2d_1_bias_mIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_conv2d_2_kernel_mIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_conv2d_2_bias_mIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_28AssignVariableOp'assignvariableop_28_adam_dense_kernel_mIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_29AssignVariableOp%assignvariableop_29_adam_dense_bias_mIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_outputs_kernel_mIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_31AssignVariableOp'assignvariableop_31_adam_outputs_bias_mIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_conv2d_kernel_vIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_33AssignVariableOp&assignvariableop_33_adam_conv2d_bias_vIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_34AssignVariableOp*assignvariableop_34_adam_conv2d_1_kernel_vIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_35AssignVariableOp(assignvariableop_35_adam_conv2d_1_bias_vIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_36AssignVariableOp*assignvariableop_36_adam_conv2d_2_kernel_vIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_conv2d_2_bias_vIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_38AssignVariableOp'assignvariableop_38_adam_dense_kernel_vIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_39AssignVariableOp%assignvariableop_39_adam_dense_bias_vIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_outputs_kernel_vIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_41AssignVariableOp'assignvariableop_41_adam_outputs_bias_vIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ы
Identity_42Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_43IdentityIdentity_42:output:0^NoOp_1*
T0*
_output_shapes
: ╪
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_43Identity_43:output:0*i
_input_shapesX
V: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412(
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
Ц
d
H__inference_module_wrapper_layer_call_and_return_conditional_losses_7283

args_0
identityX
IdentityIdentityargs_0*
T0*1
_output_shapes
:         ┤┤"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ┤┤:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameargs_0
╟
F
*__inference_random_zoom_layer_call_fn_9840

inputs
identity║
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_random_zoom_layer_call_and_return_conditional_losses_9072j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ┤┤"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ┤┤:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameinputs
╝
z
*__inference_random_flip_layer_call_fn_9627

inputs
unknown:	
identityИвStatefulPartitionedCall╘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_random_flip_layer_call_and_return_conditional_losses_8823y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ┤┤`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ┤┤: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameinputs
Ї
_
A__inference_dropout_layer_call_and_return_conditional_losses_9268

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:         @c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         @"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
═

`
A__inference_dropout_layer_call_and_return_conditional_losses_7467

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:         @Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧Ф
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>о
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ы
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:         @i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
шM
╨
E__inference_random_flip_layer_call_and_return_conditional_losses_9690

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identityИв(stateful_uniform_full_int/RngReadAndSkipi
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:i
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ы
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: b
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :Б
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ┌
(stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource)stateful_uniform_full_int/Cast/x:output:0$stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:w
-stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskП
!stateful_uniform_full_int/BitcastBitcast0stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0y
/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╧
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:У
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0_
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :С
stateful_uniform_full_intStatelessRandomUniformFullIntV2(stateful_uniform_full_int/shape:output:0,stateful_uniform_full_int/Bitcast_1:output:0*stateful_uniform_full_int/Bitcast:output:0&stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	T

zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R x
stackPack"stateful_uniform_full_int:output:0zeros_like:output:0*
N*
T0	*
_output_shapes

:d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ў
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_maskЮ
3stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:         ┤┤а
&stateless_random_flip_left_right/ShapeShape<stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
::э╧~
4stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: А
6stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:А
6stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ў
.stateless_random_flip_left_right/strided_sliceStridedSlice/stateless_random_flip_left_right/Shape:output:0=stateless_random_flip_left_right/strided_slice/stack:output:0?stateless_random_flip_left_right/strided_slice/stack_1:output:0?stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskо
?stateless_random_flip_left_right/stateless_random_uniform/shapePack7stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:В
=stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    В
=stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  А?░
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0* 
_output_shapes
::Ш
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :■
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Hstateless_random_flip_left_right/stateless_random_uniform/shape:output:0\stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0_stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:         ї
=stateless_random_flip_left_right/stateless_random_uniform/subSubFstateless_random_flip_left_right/stateless_random_uniform/max:output:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: Т
=stateless_random_flip_left_right/stateless_random_uniform/mulMul[stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Astateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:         √
9stateless_random_flip_left_right/stateless_random_uniformAddV2Astateless_random_flip_left_right/stateless_random_uniform/mul:z:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:         r
0stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :r
0stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :r
0stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :╬
.stateless_random_flip_left_right/Reshape/shapePack7stateless_random_flip_left_right/strided_slice:output:09stateless_random_flip_left_right/Reshape/shape/1:output:09stateless_random_flip_left_right/Reshape/shape/2:output:09stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:х
(stateless_random_flip_left_right/ReshapeReshape=stateless_random_flip_left_right/stateless_random_uniform:z:07stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:         Ь
&stateless_random_flip_left_right/RoundRound1stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:         y
/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:ы
*stateless_random_flip_left_right/ReverseV2	ReverseV2<stateless_random_flip_left_right/control_dependency:output:08stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:         ┤┤╚
$stateless_random_flip_left_right/mulMul*stateless_random_flip_left_right/Round:y:03stateless_random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:         ┤┤k
&stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?┬
$stateless_random_flip_left_right/subSub/stateless_random_flip_left_right/sub/x:output:0*stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:         ╤
&stateless_random_flip_left_right/mul_1Mul(stateless_random_flip_left_right/sub:z:0<stateless_random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:         ┤┤┐
$stateless_random_flip_left_right/addAddV2(stateless_random_flip_left_right/mul:z:0*stateless_random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:         ┤┤Б
IdentityIdentity(stateless_random_flip_left_right/add:z:0^NoOp*
T0*1
_output_shapes
:         ┤┤q
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ┤┤: 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameinputs
╛А
┬
I__inference_random_rotation_layer_call_and_return_conditional_losses_9824

inputs6
(stateful_uniform_rngreadandskip_resource:	
identityИвstateful_uniform/RngReadAndSkipI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
¤        j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
■        a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: d
stateful_uniform/shapePackstrided_slice:output:0*
N*
T0*
_output_shapes
:Y
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|┘ ┐Y
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|┘ ?`
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: А
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: Y
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :o
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ╢
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:n
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:м
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask}
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0p
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:в
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:Б
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0o
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :Л
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:         z
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: Ч
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*#
_output_shapes
:         А
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*#
_output_shapes
:         Z
rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?g
rotation_matrix/subSub
Cast_1:y:0rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: ^
rotation_matrix/CosCosstateful_uniform:z:0*
T0*#
_output_shapes
:         \
rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?k
rotation_matrix/sub_1Sub
Cast_1:y:0 rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: |
rotation_matrix/mulMulrotation_matrix/Cos:y:0rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:         ^
rotation_matrix/SinSinstateful_uniform:z:0*
T0*#
_output_shapes
:         \
rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?i
rotation_matrix/sub_2SubCast:y:0 rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: ~
rotation_matrix/mul_1Mulrotation_matrix/Sin:y:0rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:         ~
rotation_matrix/sub_3Subrotation_matrix/mul:z:0rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:         ~
rotation_matrix/sub_4Subrotation_matrix/sub:z:0rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:         ^
rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @П
rotation_matrix/truedivRealDivrotation_matrix/sub_4:z:0"rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:         \
rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?i
rotation_matrix/sub_5SubCast:y:0 rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: `
rotation_matrix/Sin_1Sinstateful_uniform:z:0*
T0*#
_output_shapes
:         \
rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?k
rotation_matrix/sub_6Sub
Cast_1:y:0 rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: А
rotation_matrix/mul_2Mulrotation_matrix/Sin_1:y:0rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:         `
rotation_matrix/Cos_1Cosstateful_uniform:z:0*
T0*#
_output_shapes
:         \
rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?i
rotation_matrix/sub_7SubCast:y:0 rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: А
rotation_matrix/mul_3Mulrotation_matrix/Cos_1:y:0rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:         А
rotation_matrix/addAddV2rotation_matrix/mul_2:z:0rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:         ~
rotation_matrix/sub_8Subrotation_matrix/sub_5:z:0rotation_matrix/add:z:0*
T0*#
_output_shapes
:         `
rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @У
rotation_matrix/truediv_1RealDivrotation_matrix/sub_8:z:0$rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:         g
rotation_matrix/ShapeShapestateful_uniform:z:0*
T0*
_output_shapes
::э╧m
#rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
rotation_matrix/strided_sliceStridedSlicerotation_matrix/Shape:output:0,rotation_matrix/strided_slice/stack:output:0.rotation_matrix/strided_slice/stack_1:output:0.rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
rotation_matrix/Cos_2Cosstateful_uniform:z:0*
T0*#
_output_shapes
:         v
%rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
rotation_matrix/strided_slice_1StridedSlicerotation_matrix/Cos_2:y:0.rotation_matrix/strided_slice_1/stack:output:00rotation_matrix/strided_slice_1/stack_1:output:00rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Sin_2Sinstateful_uniform:z:0*
T0*#
_output_shapes
:         v
%rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
rotation_matrix/strided_slice_2StridedSlicerotation_matrix/Sin_2:y:0.rotation_matrix/strided_slice_2/stack:output:00rotation_matrix/strided_slice_2/stack_1:output:00rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_maskv
rotation_matrix/NegNeg(rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:         v
%rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
rotation_matrix/strided_slice_3StridedSlicerotation_matrix/truediv:z:0.rotation_matrix/strided_slice_3/stack:output:00rotation_matrix/strided_slice_3/stack_1:output:00rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Sin_3Sinstateful_uniform:z:0*
T0*#
_output_shapes
:         v
%rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
rotation_matrix/strided_slice_4StridedSlicerotation_matrix/Sin_3:y:0.rotation_matrix/strided_slice_4/stack:output:00rotation_matrix/strided_slice_4/stack_1:output:00rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Cos_3Cosstateful_uniform:z:0*
T0*#
_output_shapes
:         v
%rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
rotation_matrix/strided_slice_5StridedSlicerotation_matrix/Cos_3:y:0.rotation_matrix/strided_slice_5/stack:output:00rotation_matrix/strided_slice_5/stack_1:output:00rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_maskv
%rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╪
rotation_matrix/strided_slice_6StridedSlicerotation_matrix/truediv_1:z:0.rotation_matrix/strided_slice_6/stack:output:00rotation_matrix/strided_slice_6/stack_1:output:00rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :г
rotation_matrix/zeros/packedPack&rotation_matrix/strided_slice:output:0'rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:`
rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ь
rotation_matrix/zerosFill%rotation_matrix/zeros/packed:output:0$rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:         ]
rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :О
rotation_matrix/concatConcatV2(rotation_matrix/strided_slice_1:output:0rotation_matrix/Neg:y:0(rotation_matrix/strided_slice_3:output:0(rotation_matrix/strided_slice_4:output:0(rotation_matrix/strided_slice_5:output:0(rotation_matrix/strided_slice_6:output:0rotation_matrix/zeros:output:0$rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:         S
transform/ShapeShapeinputs*
T0*
_output_shapes
::э╧g
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:я
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:Y
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    б
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputsrotation_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*1
_output_shapes
:         ┤┤*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEARТ
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:         ┤┤h
NoOpNoOp ^stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ┤┤: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameinputs
С
e
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_7254

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╛3
Х
D__inference_sequential_layer_call_and_return_conditional_losses_8013
module_wrapper_input!
module_wrapper_7974:	!
module_wrapper_7976:	!
module_wrapper_7978:	%
conv2d_7982:
conv2d_7984:'
conv2d_1_7988: 
conv2d_1_7990: '
conv2d_2_7994: @
conv2d_2_7996:@

dense_8002:АЄА

dense_8004:	А
outputs_8007:	А
outputs_8009:
identityИвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallв conv2d_2/StatefulPartitionedCallвdense/StatefulPartitionedCallвdropout/StatefulPartitionedCallв&module_wrapper/StatefulPartitionedCallвoutputs/StatefulPartitionedCallм
&module_wrapper/StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputmodule_wrapper_7974module_wrapper_7976module_wrapper_7978*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_module_wrapper_layer_call_and_return_conditional_losses_7797∙
 module_wrapper_1/PartitionedCallPartitionedCall/module_wrapper/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_7516П
conv2d/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_1/PartitionedCall:output:0conv2d_7982conv2d_7984*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_7306щ
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ZZ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_7242Т
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_7988conv2d_1_7990*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ZZ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_7324я
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         -- * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_7254Ф
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_7994conv2d_2_7996*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         --@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_7342я
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_7266ю
dropout/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_7467╪
flatten/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         АЄ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_7362∙
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dense_8002
dense_8004*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_7375Ж
outputs/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0outputs_8007outputs_8009*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_outputs_layer_call_and_return_conditional_losses_7391w
IdentityIdentity(outputs/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ║
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall'^module_wrapper/StatefulPartitionedCall ^outputs/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         ┤┤: : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2P
&module_wrapper/StatefulPartitionedCall&module_wrapper/StatefulPartitionedCall2B
outputs/StatefulPartitionedCalloutputs/StatefulPartitionedCall:g c
1
_output_shapes
:         ┤┤
.
_user_specified_namemodule_wrapper_input
┼
]
A__inference_flatten_layer_call_and_return_conditional_losses_9291

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"     y  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:         АЄZ
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:         АЄ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
к
╢
)__inference_sequential_layer_call_fn_9100
random_flip_input
unknown:	
	unknown_0:	
	unknown_1:	
identityИвStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallrandom_flip_inputunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_9091y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ┤┤`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         ┤┤: : : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
1
_output_shapes
:         ┤┤
+
_user_specified_namerandom_flip_input
╝
╪
)__inference_sequential_layer_call_fn_8102

inputs
unknown:	
	unknown_0:	
	unknown_1:	#
	unknown_2:
	unknown_3:#
	unknown_4: 
	unknown_5: #
	unknown_6: @
	unknown_7:@
	unknown_8:АЄА
	unknown_9:	А

unknown_10:	А

unknown_11:
identityИвStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_7875o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         ┤┤: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameinputs
У
a
E__inference_random_zoom_layer_call_and_return_conditional_losses_9072

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:         ┤┤"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ┤┤:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameinputs
С
e
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_7266

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ц
Ь
'__inference_conv2d_2_layer_call_fn_9242

inputs!
unknown: @
	unknown_0:@
identityИвStatefulPartitionedCall▀
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         --@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_7342w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         --@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         -- : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         -- 
 
_user_specified_nameinputs
ц
Ь
'__inference_conv2d_1_layer_call_fn_9222

inputs!
unknown: 
	unknown_0: 
identityИвStatefulPartitionedCall▀
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ZZ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_7324w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         ZZ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ZZ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         ZZ
 
_user_specified_nameinputs
Б
√
B__inference_conv2d_2_layer_call_and_return_conditional_losses_9253

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         --@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         --@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         --@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         --@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         -- : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         -- 
 
_user_specified_nameinputs
Ї
k
D__inference_sequential_layer_call_and_return_conditional_losses_9075
random_flip_input
identity╤
random_flip/PartitionedCallPartitionedCallrandom_flip_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_random_flip_layer_call_and_return_conditional_losses_9062ь
random_rotation/PartitionedCallPartitionedCall$random_flip/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_random_rotation_layer_call_and_return_conditional_losses_9067ш
random_zoom/PartitionedCallPartitionedCall(random_rotation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_random_zoom_layer_call_and_return_conditional_losses_9072v
IdentityIdentity$random_zoom/PartitionedCall:output:0*
T0*1
_output_shapes
:         ┤┤"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ┤┤:d `
1
_output_shapes
:         ┤┤
+
_user_specified_namerandom_flip_input
Й
л
)__inference_sequential_layer_call_fn_9341

inputs
unknown:	
	unknown_0:	
	unknown_1:	
identityИвStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_9091y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ┤┤`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         ┤┤: : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameinputs
З/
╙
D__inference_sequential_layer_call_and_return_conditional_losses_7398

inputs%
conv2d_7307:
conv2d_7309:'
conv2d_1_7325: 
conv2d_1_7327: '
conv2d_2_7343: @
conv2d_2_7345:@

dense_7376:АЄА

dense_7378:	А
outputs_7392:	А
outputs_7394:
identityИвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallв conv2d_2/StatefulPartitionedCallвdense/StatefulPartitionedCallвoutputs/StatefulPartitionedCall╠
module_wrapper/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_module_wrapper_layer_call_and_return_conditional_losses_7283ё
 module_wrapper_1/PartitionedCallPartitionedCall'module_wrapper/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_7293П
conv2d/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_1/PartitionedCall:output:0conv2d_7307conv2d_7309*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_7306щ
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ZZ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_7242Т
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_7325conv2d_1_7327*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ZZ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_7324я
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         -- * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_7254Ф
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_7343conv2d_2_7345*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         --@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_7342я
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_7266▐
dropout/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_7354╨
flatten/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         АЄ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_7362∙
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dense_7376
dense_7378*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_7375Ж
outputs/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0outputs_7392outputs_7394*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_outputs_layer_call_and_return_conditional_losses_7391w
IdentityIdentity(outputs/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         я
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^outputs/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:         ┤┤: : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
outputs/StatefulPartitionedCalloutputs/StatefulPartitionedCall:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameinputs
Ц
d
H__inference_module_wrapper_layer_call_and_return_conditional_losses_8491

args_0
identityX
IdentityIdentityargs_0*
T0*1
_output_shapes
:         ┤┤"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ┤┤:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameargs_0
═
I
-__inference_module_wrapper_layer_call_fn_8476

args_0
identity╜
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_module_wrapper_layer_call_and_return_conditional_losses_7283j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ┤┤"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ┤┤:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameargs_0
╤
K
/__inference_module_wrapper_1_layer_call_fn_9177

args_0
identity┐
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_7516j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ┤┤"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ┤┤:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameargs_0
╝
z
*__inference_random_zoom_layer_call_fn_9835

inputs
unknown:	
identityИвStatefulPartitionedCall╘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_random_zoom_layer_call_and_return_conditional_losses_9051y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ┤┤`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ┤┤: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameinputs
Л
∙
@__inference_conv2d_layer_call_and_return_conditional_losses_7306

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ы
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ┤┤*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ┤┤Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:         ┤┤k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:         ┤┤w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ┤┤: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameinputs
▓n
╛
E__inference_random_zoom_layer_call_and_return_conditional_losses_9051

inputs6
(stateful_uniform_rngreadandskip_resource:	
identityИвstateful_uniform/RngReadAndSkipI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
¤        j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
■        a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: Z
stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :З
stateful_uniform/shapePackstrided_slice:output:0!stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:Y
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *fff?Y
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *═╠М?`
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: А
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: Y
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :o
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ╢
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:n
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:м
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask}
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0p
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:в
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:Б
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0o
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :П
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:         z
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: Ы
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*'
_output_shapes
:         Д
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*'
_output_shapes
:         M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :П
concatConcatV2stateful_uniform:z:0stateful_uniform:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:         ^
zoom_matrix/ShapeShapeconcat:output:0*
T0*
_output_shapes
::э╧i
zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
zoom_matrix/strided_sliceStridedSlicezoom_matrix/Shape:output:0(zoom_matrix/strided_slice/stack:output:0*zoom_matrix/strided_slice/stack_1:output:0*zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?_
zoom_matrix/subSub
Cast_1:y:0zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: Z
zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @t
zoom_matrix/truedivRealDivzoom_matrix/sub:z:0zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: v
!zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            x
#zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╥
zoom_matrix/strided_slice_1StridedSliceconcat:output:0*zoom_matrix/strided_slice_1/stack:output:0,zoom_matrix/strided_slice_1/stack_1:output:0,zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskX
zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?О
zoom_matrix/sub_1Subzoom_matrix/sub_1/x:output:0$zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:         x
zoom_matrix/mulMulzoom_matrix/truediv:z:0zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:         X
zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?a
zoom_matrix/sub_2SubCast:y:0zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: \
zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @z
zoom_matrix/truediv_1RealDivzoom_matrix/sub_2:z:0 zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: v
!zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╥
zoom_matrix/strided_slice_2StridedSliceconcat:output:0*zoom_matrix/strided_slice_2/stack:output:0,zoom_matrix/strided_slice_2/stack_1:output:0,zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskX
zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?О
zoom_matrix/sub_3Subzoom_matrix/sub_3/x:output:0$zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:         |
zoom_matrix/mul_1Mulzoom_matrix/truediv_1:z:0zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:         v
!zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            x
#zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╥
zoom_matrix/strided_slice_3StridedSliceconcat:output:0*zoom_matrix/strided_slice_3/stack:output:0,zoom_matrix/strided_slice_3/stack_1:output:0,zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask\
zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ч
zoom_matrix/zeros/packedPack"zoom_matrix/strided_slice:output:0#zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:\
zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Р
zoom_matrix/zerosFill!zoom_matrix/zeros/packed:output:0 zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:         ^
zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ы
zoom_matrix/zeros_1/packedPack"zoom_matrix/strided_slice:output:0%zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:^
zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ц
zoom_matrix/zeros_1Fill#zoom_matrix/zeros_1/packed:output:0"zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:         v
!zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╥
zoom_matrix/strided_slice_4StridedSliceconcat:output:0*zoom_matrix/strided_slice_4/stack:output:0,zoom_matrix/strided_slice_4/stack_1:output:0,zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask^
zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ы
zoom_matrix/zeros_2/packedPack"zoom_matrix/strided_slice:output:0%zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:^
zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ц
zoom_matrix/zeros_2Fill#zoom_matrix/zeros_2/packed:output:0"zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:         Y
zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :╦
zoom_matrix/concatConcatV2$zoom_matrix/strided_slice_3:output:0zoom_matrix/zeros:output:0zoom_matrix/mul:z:0zoom_matrix/zeros_1:output:0$zoom_matrix/strided_slice_4:output:0zoom_matrix/mul_1:z:0zoom_matrix/zeros_2:output:0 zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:         S
transform/ShapeShapeinputs*
T0*
_output_shapes
::э╧g
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:я
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:Y
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    Э
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputszoom_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*1
_output_shapes
:         ┤┤*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEARТ
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:         ┤┤h
NoOpNoOp ^stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ┤┤: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameinputs
Ч
e
I__inference_random_rotation_layer_call_and_return_conditional_losses_9828

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:         ┤┤"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ┤┤:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameinputs
У
a
E__inference_random_flip_layer_call_and_return_conditional_losses_9062

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:         ┤┤"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ┤┤:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameinputs
╖
B
&__inference_dropout_layer_call_fn_9258

inputs
identity┤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_7354h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
╚	
є
A__inference_outputs_layer_call_and_return_conditional_losses_7391

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
┼8
Ў
D__inference_sequential_layer_call_and_return_conditional_losses_8150

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource: 6
(conv2d_1_biasadd_readvariableop_resource: A
'conv2d_2_conv2d_readvariableop_resource: @6
(conv2d_2_biasadd_readvariableop_resource:@9
$dense_matmul_readvariableop_resource:АЄА4
%dense_biasadd_readvariableop_resource:	А9
&outputs_matmul_readvariableop_resource:	А5
'outputs_biasadd_readvariableop_resource:
identityИвconv2d/BiasAdd/ReadVariableOpвconv2d/Conv2D/ReadVariableOpвconv2d_1/BiasAdd/ReadVariableOpвconv2d_1/Conv2D/ReadVariableOpвconv2d_2/BiasAdd/ReadVariableOpвconv2d_2/Conv2D/ReadVariableOpвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpвoutputs/BiasAdd/ReadVariableOpвoutputs/MatMul/ReadVariableOph
#module_wrapper_1/rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;j
%module_wrapper_1/rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Щ
 module_wrapper_1/rescaling_1/mulMulinputs,module_wrapper_1/rescaling_1/Cast/x:output:0*
T0*1
_output_shapes
:         ┤┤╗
 module_wrapper_1/rescaling_1/addAddV2$module_wrapper_1/rescaling_1/mul:z:0.module_wrapper_1/rescaling_1/Cast_1/x:output:0*
T0*1
_output_shapes
:         ┤┤К
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╟
conv2d/Conv2DConv2D$module_wrapper_1/rescaling_1/add:z:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ┤┤*
paddingSAME*
strides
А
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ф
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ┤┤h
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:         ┤┤и
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:         ZZ*
ksize
*
paddingVALID*
strides
О
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0├
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ZZ *
paddingSAME*
strides
Д
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ш
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ZZ j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         ZZ м
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:         -- *
ksize
*
paddingVALID*
strides
О
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0┼
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         --@*
paddingSAME*
strides
Д
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ш
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         --@j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:         --@м
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
x
dropout/IdentityIdentity max_pooling2d_2/MaxPool:output:0*
T0*/
_output_shapes
:         @^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"     y  Б
flatten/ReshapeReshapedropout/Identity:output:0flatten/Const:output:0*
T0*)
_output_shapes
:         АЄГ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*!
_output_shapes
:АЄА*
dtype0И
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:         АЕ
outputs/MatMul/ReadVariableOpReadVariableOp&outputs_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Л
outputs/MatMulMatMuldense/Relu:activations:0%outputs/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         В
outputs/BiasAdd/ReadVariableOpReadVariableOp'outputs_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
outputs/BiasAddBiasAddoutputs/MatMul:product:0&outputs/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         g
IdentityIdentityoutputs/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         Й
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^outputs/BiasAdd/ReadVariableOp^outputs/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:         ┤┤: : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
outputs/BiasAdd/ReadVariableOpoutputs/BiasAdd/ReadVariableOp2>
outputs/MatMul/ReadVariableOpoutputs/MatMul/ReadVariableOp:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameinputs
┤
J
.__inference_max_pooling2d_2_layer_call_fn_7272

inputs
identity╫
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_7266Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╤
K
/__inference_module_wrapper_1_layer_call_fn_9172

args_0
identity┐
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_7293j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ┤┤"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ┤┤:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameargs_0
▌
f
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_7293

args_0
identityW
rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;Y
rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    w
rescaling_1/mulMulargs_0rescaling_1/Cast/x:output:0*
T0*1
_output_shapes
:         ┤┤И
rescaling_1/addAddV2rescaling_1/mul:z:0rescaling_1/Cast_1/x:output:0*
T0*1
_output_shapes
:         ┤┤e
IdentityIdentityrescaling_1/add:z:0*
T0*1
_output_shapes
:         ┤┤"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ┤┤:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameargs_0
ъ
Ъ
%__inference_conv2d_layer_call_fn_9202

inputs!
unknown:
	unknown_0:
identityИвStatefulPartitionedCall▀
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_7306y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ┤┤`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ┤┤: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameinputs
┐
Ф
&__inference_outputs_layer_call_fn_9320

inputs
unknown:	А
	unknown_0:
identityИвStatefulPartitionedCall╓
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_outputs_layer_call_and_return_conditional_losses_7391o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
═

`
A__inference_dropout_layer_call_and_return_conditional_losses_9280

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:         @Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧Ф
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>о
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ы
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:         @i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
╬C
╗	
__inference__wrapped_model_7236
module_wrapper_inputJ
0sequential_conv2d_conv2d_readvariableop_resource:?
1sequential_conv2d_biasadd_readvariableop_resource:L
2sequential_conv2d_1_conv2d_readvariableop_resource: A
3sequential_conv2d_1_biasadd_readvariableop_resource: L
2sequential_conv2d_2_conv2d_readvariableop_resource: @A
3sequential_conv2d_2_biasadd_readvariableop_resource:@D
/sequential_dense_matmul_readvariableop_resource:АЄА?
0sequential_dense_biasadd_readvariableop_resource:	АD
1sequential_outputs_matmul_readvariableop_resource:	А@
2sequential_outputs_biasadd_readvariableop_resource:
identityИв(sequential/conv2d/BiasAdd/ReadVariableOpв'sequential/conv2d/Conv2D/ReadVariableOpв*sequential/conv2d_1/BiasAdd/ReadVariableOpв)sequential/conv2d_1/Conv2D/ReadVariableOpв*sequential/conv2d_2/BiasAdd/ReadVariableOpв)sequential/conv2d_2/Conv2D/ReadVariableOpв'sequential/dense/BiasAdd/ReadVariableOpв&sequential/dense/MatMul/ReadVariableOpв)sequential/outputs/BiasAdd/ReadVariableOpв(sequential/outputs/MatMul/ReadVariableOps
.sequential/module_wrapper_1/rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;u
0sequential/module_wrapper_1/rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ╜
+sequential/module_wrapper_1/rescaling_1/mulMulmodule_wrapper_input7sequential/module_wrapper_1/rescaling_1/Cast/x:output:0*
T0*1
_output_shapes
:         ┤┤▄
+sequential/module_wrapper_1/rescaling_1/addAddV2/sequential/module_wrapper_1/rescaling_1/mul:z:09sequential/module_wrapper_1/rescaling_1/Cast_1/x:output:0*
T0*1
_output_shapes
:         ┤┤а
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ш
sequential/conv2d/Conv2DConv2D/sequential/module_wrapper_1/rescaling_1/add:z:0/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ┤┤*
paddingSAME*
strides
Ц
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╡
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ┤┤~
sequential/conv2d/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:         ┤┤╛
 sequential/max_pooling2d/MaxPoolMaxPool$sequential/conv2d/Relu:activations:0*/
_output_shapes
:         ZZ*
ksize
*
paddingVALID*
strides
д
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ф
sequential/conv2d_1/Conv2DConv2D)sequential/max_pooling2d/MaxPool:output:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ZZ *
paddingSAME*
strides
Ъ
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0╣
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ZZ А
sequential/conv2d_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         ZZ ┬
"sequential/max_pooling2d_1/MaxPoolMaxPool&sequential/conv2d_1/Relu:activations:0*/
_output_shapes
:         -- *
ksize
*
paddingVALID*
strides
д
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0ц
sequential/conv2d_2/Conv2DConv2D+sequential/max_pooling2d_1/MaxPool:output:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         --@*
paddingSAME*
strides
Ъ
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╣
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         --@А
sequential/conv2d_2/ReluRelu$sequential/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:         --@┬
"sequential/max_pooling2d_2/MaxPoolMaxPool&sequential/conv2d_2/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
О
sequential/dropout/IdentityIdentity+sequential/max_pooling2d_2/MaxPool:output:0*
T0*/
_output_shapes
:         @i
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"     y  в
sequential/flatten/ReshapeReshape$sequential/dropout/Identity:output:0!sequential/flatten/Const:output:0*
T0*)
_output_shapes
:         АЄЩ
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*!
_output_shapes
:АЄА*
dtype0й
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АХ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0к
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:         АЫ
(sequential/outputs/MatMul/ReadVariableOpReadVariableOp1sequential_outputs_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0м
sequential/outputs/MatMulMatMul#sequential/dense/Relu:activations:00sequential/outputs/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ш
)sequential/outputs/BiasAdd/ReadVariableOpReadVariableOp2sequential_outputs_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0п
sequential/outputs/BiasAddBiasAdd#sequential/outputs/MatMul:product:01sequential/outputs/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
IdentityIdentity#sequential/outputs/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         ў
NoOpNoOp)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp+^sequential/conv2d_2/BiasAdd/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/outputs/BiasAdd/ReadVariableOp)^sequential/outputs/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:         ┤┤: : : : : : : : : : 2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2X
*sequential/conv2d_2/BiasAdd/ReadVariableOp*sequential/conv2d_2/BiasAdd/ReadVariableOp2V
)sequential/conv2d_2/Conv2D/ReadVariableOp)sequential/conv2d_2/Conv2D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/outputs/BiasAdd/ReadVariableOp)sequential/outputs/BiasAdd/ReadVariableOp2T
(sequential/outputs/MatMul/ReadVariableOp(sequential/outputs/MatMul/ReadVariableOp:g c
1
_output_shapes
:         ┤┤
.
_user_specified_namemodule_wrapper_input
ц
ц
)__inference_sequential_layer_call_fn_7935
module_wrapper_input
unknown:	
	unknown_0:	
	unknown_1:	#
	unknown_2:
	unknown_3:#
	unknown_4: 
	unknown_5: #
	unknown_6: @
	unknown_7:@
	unknown_8:АЄА
	unknown_9:	А

unknown_10:	А

unknown_11:
identityИвStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_7875o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         ┤┤: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
1
_output_shapes
:         ┤┤
.
_user_specified_namemodule_wrapper_input
Ї
_
A__inference_dropout_layer_call_and_return_conditional_losses_7354

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:         @c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         @"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
├

Л
)__inference_sequential_layer_call_fn_8071

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@
	unknown_5:АЄА
	unknown_6:	А
	unknown_7:	А
	unknown_8:
identityИвStatefulPartitionedCall┴
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_7398o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:         ┤┤: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameinputs
шM
╨
E__inference_random_flip_layer_call_and_return_conditional_losses_8823

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identityИв(stateful_uniform_full_int/RngReadAndSkipi
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:i
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ы
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: b
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :Б
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ┌
(stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource)stateful_uniform_full_int/Cast/x:output:0$stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:w
-stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskП
!stateful_uniform_full_int/BitcastBitcast0stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0y
/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╧
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:У
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0_
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :С
stateful_uniform_full_intStatelessRandomUniformFullIntV2(stateful_uniform_full_int/shape:output:0,stateful_uniform_full_int/Bitcast_1:output:0*stateful_uniform_full_int/Bitcast:output:0&stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	T

zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R x
stackPack"stateful_uniform_full_int:output:0zeros_like:output:0*
N*
T0	*
_output_shapes

:d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ў
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_maskЮ
3stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:         ┤┤а
&stateless_random_flip_left_right/ShapeShape<stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
::э╧~
4stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: А
6stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:А
6stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ў
.stateless_random_flip_left_right/strided_sliceStridedSlice/stateless_random_flip_left_right/Shape:output:0=stateless_random_flip_left_right/strided_slice/stack:output:0?stateless_random_flip_left_right/strided_slice/stack_1:output:0?stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskо
?stateless_random_flip_left_right/stateless_random_uniform/shapePack7stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:В
=stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    В
=stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  А?░
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0* 
_output_shapes
::Ш
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :■
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Hstateless_random_flip_left_right/stateless_random_uniform/shape:output:0\stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0_stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:         ї
=stateless_random_flip_left_right/stateless_random_uniform/subSubFstateless_random_flip_left_right/stateless_random_uniform/max:output:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: Т
=stateless_random_flip_left_right/stateless_random_uniform/mulMul[stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Astateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:         √
9stateless_random_flip_left_right/stateless_random_uniformAddV2Astateless_random_flip_left_right/stateless_random_uniform/mul:z:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:         r
0stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :r
0stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :r
0stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :╬
.stateless_random_flip_left_right/Reshape/shapePack7stateless_random_flip_left_right/strided_slice:output:09stateless_random_flip_left_right/Reshape/shape/1:output:09stateless_random_flip_left_right/Reshape/shape/2:output:09stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:х
(stateless_random_flip_left_right/ReshapeReshape=stateless_random_flip_left_right/stateless_random_uniform:z:07stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:         Ь
&stateless_random_flip_left_right/RoundRound1stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:         y
/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:ы
*stateless_random_flip_left_right/ReverseV2	ReverseV2<stateless_random_flip_left_right/control_dependency:output:08stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:         ┤┤╚
$stateless_random_flip_left_right/mulMul*stateless_random_flip_left_right/Round:y:03stateless_random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:         ┤┤k
&stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?┬
$stateless_random_flip_left_right/subSub/stateless_random_flip_left_right/sub/x:output:0*stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:         ╤
&stateless_random_flip_left_right/mul_1Mul(stateless_random_flip_left_right/sub:z:0<stateless_random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:         ┤┤┐
$stateless_random_flip_left_right/addAddV2(stateless_random_flip_left_right/mul:z:0*stateless_random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:         ┤┤Б
IdentityIdentity(stateless_random_flip_left_right/add:z:0^NoOp*
T0*1
_output_shapes
:         ┤┤q
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ┤┤: 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameinputs
ц
P
)__inference_sequential_layer_call_fn_9112
random_flip_input
identity─
PartitionedCallPartitionedCallrandom_flip_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_9109j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ┤┤"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ┤┤:d `
1
_output_shapes
:         ┤┤
+
_user_specified_namerandom_flip_input
С
п
-__inference_module_wrapper_layer_call_fn_8487

args_0
unknown:	
	unknown_0:	
	unknown_1:	
identityИвStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_module_wrapper_layer_call_and_return_conditional_losses_7797y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ┤┤`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         ┤┤: : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameargs_0
ж

Ї
?__inference_dense_layer_call_and_return_conditional_losses_7375

inputs3
matmul_readvariableop_resource:АЄА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpw
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:АЄА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         АЄ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:         АЄ
 
_user_specified_nameinputs
╙
`
D__inference_sequential_layer_call_and_return_conditional_losses_9109

inputs
identity╞
random_flip/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_random_flip_layer_call_and_return_conditional_losses_9062ь
random_rotation/PartitionedCallPartitionedCall$random_flip/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_random_rotation_layer_call_and_return_conditional_losses_9067ш
random_zoom/PartitionedCallPartitionedCall(random_rotation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ┤┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_random_zoom_layer_call_and_return_conditional_losses_9072v
IdentityIdentity$random_zoom/PartitionedCall:output:0*
T0*1
_output_shapes
:         ┤┤"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ┤┤:Y U
1
_output_shapes
:         ┤┤
 
_user_specified_nameinputs"є
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╬
serving_default║
_
module_wrapper_inputG
&serving_default_module_wrapper_input:0         ┤┤;
outputs0
StatefulPartitionedCall:0         tensorflow/serving/predict:Эо
Ж
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
layer_with_weights-4
layer-11
	variables
trainable_variables
regularization_losses
	keras_api
*&call_and_return_all_conditional_losses
__call__
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
▓
	variables
regularization_losses
trainable_variables
	keras_api
__call__
*&call_and_return_all_conditional_losses
_module"
_tf_keras_layer
▓
	variables
regularization_losses
trainable_variables
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
#_module"
_tf_keras_layer
╗
$	variables
%regularization_losses
&trainable_variables
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

*kernel
+bias"
_tf_keras_layer
е
,	variables
-regularization_losses
.trainable_variables
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses"
_tf_keras_layer
╗
2	variables
3regularization_losses
4trainable_variables
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses

8kernel
9bias"
_tf_keras_layer
е
:	variables
;regularization_losses
<trainable_variables
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses"
_tf_keras_layer
╗
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

Fkernel
Gbias"
_tf_keras_layer
е
H	variables
Iregularization_losses
Jtrainable_variables
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses"
_tf_keras_layer
е
N	variables
Oregularization_losses
Ptrainable_variables
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses"
_tf_keras_layer
е
T	variables
Uregularization_losses
Vtrainable_variables
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses"
_tf_keras_layer
╗
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses

`kernel
abias"
_tf_keras_layer
╗
b	variables
cregularization_losses
dtrainable_variables
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

hkernel
ibias"
_tf_keras_layer
f
*0
+1
82
93
F4
G5
`6
a7
h8
i9"
trackable_list_wrapper
f
*0
+1
82
93
F4
G5
`6
a7
h8
i9"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
jlayer_metrics
	variables

klayers
lnon_trainable_variables
mlayer_regularization_losses
nmetrics
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
╗
otrace_0
ptrace_1
qtrace_2
rtrace_32╨
D__inference_sequential_layer_call_and_return_conditional_losses_8150
D__inference_sequential_layer_call_and_return_conditional_losses_8471
D__inference_sequential_layer_call_and_return_conditional_losses_7971
D__inference_sequential_layer_call_and_return_conditional_losses_8013╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zotrace_0zptrace_1zqtrace_2zrtrace_3
╧
strace_0
ttrace_1
utrace_2
vtrace_32ф
)__inference_sequential_layer_call_fn_7421
)__inference_sequential_layer_call_fn_8071
)__inference_sequential_layer_call_fn_8102
)__inference_sequential_layer_call_fn_7935╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zstrace_0zttrace_1zutrace_2zvtrace_3
Ф
wtrace_02ў
__inference__wrapped_model_7236╙
С▓Н
FullArgSpec
argsЪ

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *=в:
8К5
module_wrapper_input         ┤┤zwtrace_0
й
xiter

ybeta_1

zbeta_2
	{decay
|learning_rate*m║+m╗8m╝9m╜Fm╛Gm┐`m└am┴hm┬im├*v─+v┼8v╞9v╟Fv╚Gv╔`v╩av╦hv╠iv═"
tf_deprecated_optimizer
,
}serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
~layer_metrics
	variables
non_trainable_variables
regularization_losses
 Аlayer_regularization_losses
Бmetrics
trainable_variables
Вlayers
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
╤
Гtrace_0
Дtrace_12Ц
-__inference_module_wrapper_layer_call_fn_8476
-__inference_module_wrapper_layer_call_fn_8487╡
о▓к
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 zГtrace_0zДtrace_1
З
Еtrace_0
Жtrace_12╠
H__inference_module_wrapper_layer_call_and_return_conditional_losses_8491
H__inference_module_wrapper_layer_call_and_return_conditional_losses_8761╡
о▓к
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 zЕtrace_0zЖtrace_1
┌
Зlayer-0
Иlayer-1
Йlayer-2
К	variables
Лtrainable_variables
Мregularization_losses
Н	keras_api
О__call__
+П&call_and_return_all_conditional_losses"
_tf_keras_sequential
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Рlayer_metrics
	variables
Сnon_trainable_variables
regularization_losses
 Тlayer_regularization_losses
Уmetrics
trainable_variables
Фlayers
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
╒
Хtrace_0
Цtrace_12Ъ
/__inference_module_wrapper_1_layer_call_fn_9172
/__inference_module_wrapper_1_layer_call_fn_9177╡
о▓к
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 zХtrace_0zЦtrace_1
Л
Чtrace_0
Шtrace_12╨
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_9185
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_9193╡
о▓к
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 zЧtrace_0zШtrace_1
л
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Ь	keras_api
Э__call__
+Ю&call_and_return_all_conditional_losses"
_tf_keras_layer
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
▓
Яlayer_metrics
$	variables
аnon_trainable_variables
%regularization_losses
 бlayer_regularization_losses
вmetrics
&trainable_variables
гlayers
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
с
дtrace_02┬
%__inference_conv2d_layer_call_fn_9202Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zдtrace_0
№
еtrace_02▌
@__inference_conv2d_layer_call_and_return_conditional_losses_9213Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zеtrace_0
':%2conv2d/kernel
:2conv2d/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
жlayer_metrics
,	variables
зnon_trainable_variables
-regularization_losses
 иlayer_regularization_losses
йmetrics
.trainable_variables
кlayers
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
ж
лtrace_02З
,__inference_max_pooling2d_layer_call_fn_7248╓
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    zлtrace_0
┴
мtrace_02в
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_7242╓
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    zмtrace_0
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
▓
нlayer_metrics
2	variables
оnon_trainable_variables
3regularization_losses
 пlayer_regularization_losses
░metrics
4trainable_variables
▒layers
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
у
▓trace_02─
'__inference_conv2d_1_layer_call_fn_9222Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z▓trace_0
■
│trace_02▀
B__inference_conv2d_1_layer_call_and_return_conditional_losses_9233Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z│trace_0
):' 2conv2d_1/kernel
: 2conv2d_1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
┤layer_metrics
:	variables
╡non_trainable_variables
;regularization_losses
 ╢layer_regularization_losses
╖metrics
<trainable_variables
╕layers
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
и
╣trace_02Й
.__inference_max_pooling2d_1_layer_call_fn_7260╓
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    z╣trace_0
├
║trace_02д
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_7254╓
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    z║trace_0
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
▓
╗layer_metrics
@	variables
╝non_trainable_variables
Aregularization_losses
 ╜layer_regularization_losses
╛metrics
Btrainable_variables
┐layers
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
у
└trace_02─
'__inference_conv2d_2_layer_call_fn_9242Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z└trace_0
■
┴trace_02▀
B__inference_conv2d_2_layer_call_and_return_conditional_losses_9253Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┴trace_0
):' @2conv2d_2/kernel
:@2conv2d_2/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
┬layer_metrics
H	variables
├non_trainable_variables
Iregularization_losses
 ─layer_regularization_losses
┼metrics
Jtrainable_variables
╞layers
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
и
╟trace_02Й
.__inference_max_pooling2d_2_layer_call_fn_7272╓
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    z╟trace_0
├
╚trace_02д
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_7266╓
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    z╚trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╔layer_metrics
N	variables
╩non_trainable_variables
Oregularization_losses
 ╦layer_regularization_losses
╠metrics
Ptrainable_variables
═layers
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
╖
╬trace_0
╧trace_12№
&__inference_dropout_layer_call_fn_9258
&__inference_dropout_layer_call_fn_9263й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╬trace_0z╧trace_1
э
╨trace_0
╤trace_12▓
A__inference_dropout_layer_call_and_return_conditional_losses_9268
A__inference_dropout_layer_call_and_return_conditional_losses_9280й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╨trace_0z╤trace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╥layer_metrics
T	variables
╙non_trainable_variables
Uregularization_losses
 ╘layer_regularization_losses
╒metrics
Vtrainable_variables
╓layers
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
т
╫trace_02├
&__inference_flatten_layer_call_fn_9285Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╫trace_0
¤
╪trace_02▐
A__inference_flatten_layer_call_and_return_conditional_losses_9291Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╪trace_0
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
▓
┘layer_metrics
Z	variables
┌non_trainable_variables
[regularization_losses
 █layer_regularization_losses
▄metrics
\trainable_variables
▌layers
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
р
▐trace_02┴
$__inference_dense_layer_call_fn_9300Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z▐trace_0
√
▀trace_02▄
?__inference_dense_layer_call_and_return_conditional_losses_9311Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z▀trace_0
!:АЄА2dense/kernel
:А2
dense/bias
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
▓
рlayer_metrics
b	variables
сnon_trainable_variables
cregularization_losses
 тlayer_regularization_losses
уmetrics
dtrainable_variables
фlayers
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
т
хtrace_02├
&__inference_outputs_layer_call_fn_9320Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zхtrace_0
¤
цtrace_02▐
A__inference_outputs_layer_call_and_return_conditional_losses_9330Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zцtrace_0
!:	А2outputs/kernel
:2outputs/bias
 "
trackable_dict_wrapper
v
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
10
11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
ч0
ш1"
trackable_list_wrapper
ЛBИ
D__inference_sequential_layer_call_and_return_conditional_losses_8150inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЛBИ
D__inference_sequential_layer_call_and_return_conditional_losses_8471inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЩBЦ
D__inference_sequential_layer_call_and_return_conditional_losses_7971module_wrapper_input"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЩBЦ
D__inference_sequential_layer_call_and_return_conditional_losses_8013module_wrapper_input"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
■B√
)__inference_sequential_layer_call_fn_7421module_wrapper_input"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЁBэ
)__inference_sequential_layer_call_fn_8071inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЁBэ
)__inference_sequential_layer_call_fn_8102inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
■B√
)__inference_sequential_layer_call_fn_7935module_wrapper_input"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ТBП
__inference__wrapped_model_7236module_wrapper_input"╙
С▓Н
FullArgSpec
argsЪ

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *=в:
8К5
module_wrapper_input         ┤┤
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
╓B╙
"__inference_signature_wrapper_8046module_wrapper_input"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ЇBё
-__inference_module_wrapper_layer_call_fn_8476args_0"╡
о▓к
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
ЇBё
-__inference_module_wrapper_layer_call_fn_8487args_0"╡
о▓к
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
ПBМ
H__inference_module_wrapper_layer_call_and_return_conditional_losses_8491args_0"╡
о▓к
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
ПBМ
H__inference_module_wrapper_layer_call_and_return_conditional_losses_8761args_0"╡
о▓к
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
├
щ	variables
ъtrainable_variables
ыregularization_losses
ь	keras_api
э__call__
+ю&call_and_return_all_conditional_losses
я_random_generator"
_tf_keras_layer
├
Ё	variables
ёtrainable_variables
Єregularization_losses
є	keras_api
Ї__call__
+ї&call_and_return_all_conditional_losses
Ў_random_generator"
_tf_keras_layer
├
ў	variables
°trainable_variables
∙regularization_losses
·	keras_api
√__call__
+№&call_and_return_all_conditional_losses
¤_random_generator"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
■non_trainable_variables
 layers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
К	variables
Лtrainable_variables
Мregularization_losses
О__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
╫
Гtrace_0
Дtrace_1
Еtrace_2
Жtrace_32ф
)__inference_sequential_layer_call_fn_9100
)__inference_sequential_layer_call_fn_9112
)__inference_sequential_layer_call_fn_9341
)__inference_sequential_layer_call_fn_9346╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zГtrace_0zДtrace_1zЕtrace_2zЖtrace_3
├
Зtrace_0
Иtrace_1
Йtrace_2
Кtrace_32╨
D__inference_sequential_layer_call_and_return_conditional_losses_9056
D__inference_sequential_layer_call_and_return_conditional_losses_9075
D__inference_sequential_layer_call_and_return_conditional_losses_9616
D__inference_sequential_layer_call_and_return_conditional_losses_9620╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЗtrace_0zИtrace_1zЙtrace_2zКtrace_3
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ЎBє
/__inference_module_wrapper_1_layer_call_fn_9172args_0"╡
о▓к
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
ЎBє
/__inference_module_wrapper_1_layer_call_fn_9177args_0"╡
о▓к
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
СBО
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_9185args_0"╡
о▓к
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
СBО
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_9193args_0"╡
о▓к
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Лnon_trainable_variables
Мlayers
Нmetrics
 Оlayer_regularization_losses
Пlayer_metrics
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Э__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╧B╠
%__inference_conv2d_layer_call_fn_9202inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ъBч
@__inference_conv2d_layer_call_and_return_conditional_losses_9213inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ФBС
,__inference_max_pooling2d_layer_call_fn_7248inputs"╓
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
пBм
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_7242inputs"╓
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╤B╬
'__inference_conv2d_1_layer_call_fn_9222inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ьBщ
B__inference_conv2d_1_layer_call_and_return_conditional_losses_9233inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ЦBУ
.__inference_max_pooling2d_1_layer_call_fn_7260inputs"╓
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
▒Bо
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_7254inputs"╓
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╤B╬
'__inference_conv2d_2_layer_call_fn_9242inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ьBщ
B__inference_conv2d_2_layer_call_and_return_conditional_losses_9253inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ЦBУ
.__inference_max_pooling2d_2_layer_call_fn_7272inputs"╓
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
▒Bо
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_7266inputs"╓
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
сB▐
&__inference_dropout_layer_call_fn_9258inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
сB▐
&__inference_dropout_layer_call_fn_9263inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
№B∙
A__inference_dropout_layer_call_and_return_conditional_losses_9268inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
№B∙
A__inference_dropout_layer_call_and_return_conditional_losses_9280inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╨B═
&__inference_flatten_layer_call_fn_9285inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ыBш
A__inference_flatten_layer_call_and_return_conditional_losses_9291inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╬B╦
$__inference_dense_layer_call_fn_9300inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
щBц
?__inference_dense_layer_call_and_return_conditional_losses_9311inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╨B═
&__inference_outputs_layer_call_fn_9320inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ыBш
A__inference_outputs_layer_call_and_return_conditional_losses_9330inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
R
Р	variables
С	keras_api

Тtotal

Уcount"
_tf_keras_metric
c
Ф	variables
Х	keras_api

Цtotal

Чcount
Ш
_fn_kwargs"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Щnon_trainable_variables
Ъlayers
Ыmetrics
 Ьlayer_regularization_losses
Эlayer_metrics
щ	variables
ъtrainable_variables
ыregularization_losses
э__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses"
_generic_user_object
┐
Юtrace_0
Яtrace_12Д
*__inference_random_flip_layer_call_fn_9627
*__inference_random_flip_layer_call_fn_9632й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЮtrace_0zЯtrace_1
ї
аtrace_0
бtrace_12║
E__inference_random_flip_layer_call_and_return_conditional_losses_9690
E__inference_random_flip_layer_call_and_return_conditional_losses_9694й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zаtrace_0zбtrace_1
/
в
_generator"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
гnon_trainable_variables
дlayers
еmetrics
 жlayer_regularization_losses
зlayer_metrics
Ё	variables
ёtrainable_variables
Єregularization_losses
Ї__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses"
_generic_user_object
╟
иtrace_0
йtrace_12М
.__inference_random_rotation_layer_call_fn_9701
.__inference_random_rotation_layer_call_fn_9706й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zиtrace_0zйtrace_1
¤
кtrace_0
лtrace_12┬
I__inference_random_rotation_layer_call_and_return_conditional_losses_9824
I__inference_random_rotation_layer_call_and_return_conditional_losses_9828й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zкtrace_0zлtrace_1
/
м
_generator"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
нnon_trainable_variables
оlayers
пmetrics
 ░layer_regularization_losses
▒layer_metrics
ў	variables
°trainable_variables
∙regularization_losses
√__call__
+№&call_and_return_all_conditional_losses
'№"call_and_return_conditional_losses"
_generic_user_object
┐
▓trace_0
│trace_12Д
*__inference_random_zoom_layer_call_fn_9835
*__inference_random_zoom_layer_call_fn_9840й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z▓trace_0z│trace_1
ї
┤trace_0
╡trace_12║
E__inference_random_zoom_layer_call_and_return_conditional_losses_9942
E__inference_random_zoom_layer_call_and_return_conditional_losses_9946й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┤trace_0z╡trace_1
/
╢
_generator"
_generic_user_object
 "
trackable_list_wrapper
8
З0
И1
Й2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
√B°
)__inference_sequential_layer_call_fn_9100random_flip_input"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
)__inference_sequential_layer_call_fn_9112random_flip_input"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЁBэ
)__inference_sequential_layer_call_fn_9341inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЁBэ
)__inference_sequential_layer_call_fn_9346inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЦBУ
D__inference_sequential_layer_call_and_return_conditional_losses_9056random_flip_input"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЦBУ
D__inference_sequential_layer_call_and_return_conditional_losses_9075random_flip_input"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЛBИ
D__inference_sequential_layer_call_and_return_conditional_losses_9616inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЛBИ
D__inference_sequential_layer_call_and_return_conditional_losses_9620inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
0
Т0
У1"
trackable_list_wrapper
.
Р	variables"
_generic_user_object
:  (2total
:  (2count
0
Ц0
Ч1"
trackable_list_wrapper
.
Ф	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
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
хBт
*__inference_random_flip_layer_call_fn_9627inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
хBт
*__inference_random_flip_layer_call_fn_9632inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
АB¤
E__inference_random_flip_layer_call_and_return_conditional_losses_9690inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
АB¤
E__inference_random_flip_layer_call_and_return_conditional_losses_9694inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
/
╖
_state_var"
_generic_user_object
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
щBц
.__inference_random_rotation_layer_call_fn_9701inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
щBц
.__inference_random_rotation_layer_call_fn_9706inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ДBБ
I__inference_random_rotation_layer_call_and_return_conditional_losses_9824inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ДBБ
I__inference_random_rotation_layer_call_and_return_conditional_losses_9828inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
/
╕
_state_var"
_generic_user_object
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
хBт
*__inference_random_zoom_layer_call_fn_9835inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
хBт
*__inference_random_zoom_layer_call_fn_9840inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
АB¤
E__inference_random_zoom_layer_call_and_return_conditional_losses_9942inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
АB¤
E__inference_random_zoom_layer_call_and_return_conditional_losses_9946inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
/
╣
_state_var"
_generic_user_object
 :	2random_flip/StateVar
$:"	2random_rotation/StateVar
 :	2random_zoom/StateVar
,:*2Adam/conv2d/kernel/m
:2Adam/conv2d/bias/m
.:, 2Adam/conv2d_1/kernel/m
 : 2Adam/conv2d_1/bias/m
.:, @2Adam/conv2d_2/kernel/m
 :@2Adam/conv2d_2/bias/m
&:$АЄА2Adam/dense/kernel/m
:А2Adam/dense/bias/m
&:$	А2Adam/outputs/kernel/m
:2Adam/outputs/bias/m
,:*2Adam/conv2d/kernel/v
:2Adam/conv2d/bias/v
.:, 2Adam/conv2d_1/kernel/v
 : 2Adam/conv2d_1/bias/v
.:, @2Adam/conv2d_2/kernel/v
 :@2Adam/conv2d_2/bias/v
&:$АЄА2Adam/dense/kernel/v
:А2Adam/dense/bias/v
&:$	А2Adam/outputs/kernel/v
:2Adam/outputs/bias/vм
__inference__wrapped_model_7236И
*+89FG`ahiGвD
=в:
8К5
module_wrapper_input         ┤┤
к "1к.
,
outputs!К
outputs         ╣
B__inference_conv2d_1_layer_call_and_return_conditional_losses_9233s897в4
-в*
(К%
inputs         ZZ
к "4в1
*К'
tensor_0         ZZ 
Ъ У
'__inference_conv2d_1_layer_call_fn_9222h897в4
-в*
(К%
inputs         ZZ
к ")К&
unknown         ZZ ╣
B__inference_conv2d_2_layer_call_and_return_conditional_losses_9253sFG7в4
-в*
(К%
inputs         -- 
к "4в1
*К'
tensor_0         --@
Ъ У
'__inference_conv2d_2_layer_call_fn_9242hFG7в4
-в*
(К%
inputs         -- 
к ")К&
unknown         --@╗
@__inference_conv2d_layer_call_and_return_conditional_losses_9213w*+9в6
/в,
*К'
inputs         ┤┤
к "6в3
,К)
tensor_0         ┤┤
Ъ Х
%__inference_conv2d_layer_call_fn_9202l*+9в6
/в,
*К'
inputs         ┤┤
к "+К(
unknown         ┤┤й
?__inference_dense_layer_call_and_return_conditional_losses_9311f`a1в.
'в$
"К
inputs         АЄ
к "-в*
#К 
tensor_0         А
Ъ Г
$__inference_dense_layer_call_fn_9300[`a1в.
'в$
"К
inputs         АЄ
к ""К
unknown         А╕
A__inference_dropout_layer_call_and_return_conditional_losses_9268s;в8
1в.
(К%
inputs         @
p 
к "4в1
*К'
tensor_0         @
Ъ ╕
A__inference_dropout_layer_call_and_return_conditional_losses_9280s;в8
1в.
(К%
inputs         @
p
к "4в1
*К'
tensor_0         @
Ъ Т
&__inference_dropout_layer_call_fn_9258h;в8
1в.
(К%
inputs         @
p 
к ")К&
unknown         @Т
&__inference_dropout_layer_call_fn_9263h;в8
1в.
(К%
inputs         @
p
к ")К&
unknown         @о
A__inference_flatten_layer_call_and_return_conditional_losses_9291i7в4
-в*
(К%
inputs         @
к ".в+
$К!
tensor_0         АЄ
Ъ И
&__inference_flatten_layer_call_fn_9285^7в4
-в*
(К%
inputs         @
к "#К 
unknown         АЄє
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_7254еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ═
.__inference_max_pooling2d_1_layer_call_fn_7260ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    є
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_7266еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ═
.__inference_max_pooling2d_2_layer_call_fn_7272ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    ё
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_7242еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╦
,__inference_max_pooling2d_layer_call_fn_7248ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    ╥
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_9185ГIвF
/в,
*К'
args_0         ┤┤
к

trainingp "6в3
,К)
tensor_0         ┤┤
Ъ ╥
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_9193ГIвF
/в,
*К'
args_0         ┤┤
к

trainingp"6в3
,К)
tensor_0         ┤┤
Ъ л
/__inference_module_wrapper_1_layer_call_fn_9172xIвF
/в,
*К'
args_0         ┤┤
к

trainingp "+К(
unknown         ┤┤л
/__inference_module_wrapper_1_layer_call_fn_9177xIвF
/в,
*К'
args_0         ┤┤
к

trainingp"+К(
unknown         ┤┤╨
H__inference_module_wrapper_layer_call_and_return_conditional_losses_8491ГIвF
/в,
*К'
args_0         ┤┤
к

trainingp "6в3
,К)
tensor_0         ┤┤
Ъ ╪
H__inference_module_wrapper_layer_call_and_return_conditional_losses_8761Л╖╕╣IвF
/в,
*К'
args_0         ┤┤
к

trainingp"6в3
,К)
tensor_0         ┤┤
Ъ й
-__inference_module_wrapper_layer_call_fn_8476xIвF
/в,
*К'
args_0         ┤┤
к

trainingp "+К(
unknown         ┤┤▓
-__inference_module_wrapper_layer_call_fn_8487А╖╕╣IвF
/в,
*К'
args_0         ┤┤
к

trainingp"+К(
unknown         ┤┤й
A__inference_outputs_layer_call_and_return_conditional_losses_9330dhi0в-
&в#
!К
inputs         А
к ",в)
"К
tensor_0         
Ъ Г
&__inference_outputs_layer_call_fn_9320Yhi0в-
&в#
!К
inputs         А
к "!К
unknown         ─
E__inference_random_flip_layer_call_and_return_conditional_losses_9690{╖=в:
3в0
*К'
inputs         ┤┤
p
к "6в3
,К)
tensor_0         ┤┤
Ъ └
E__inference_random_flip_layer_call_and_return_conditional_losses_9694w=в:
3в0
*К'
inputs         ┤┤
p 
к "6в3
,К)
tensor_0         ┤┤
Ъ Ю
*__inference_random_flip_layer_call_fn_9627p╖=в:
3в0
*К'
inputs         ┤┤
p
к "+К(
unknown         ┤┤Ъ
*__inference_random_flip_layer_call_fn_9632l=в:
3в0
*К'
inputs         ┤┤
p 
к "+К(
unknown         ┤┤╚
I__inference_random_rotation_layer_call_and_return_conditional_losses_9824{╕=в:
3в0
*К'
inputs         ┤┤
p
к "6в3
,К)
tensor_0         ┤┤
Ъ ─
I__inference_random_rotation_layer_call_and_return_conditional_losses_9828w=в:
3в0
*К'
inputs         ┤┤
p 
к "6в3
,К)
tensor_0         ┤┤
Ъ в
.__inference_random_rotation_layer_call_fn_9701p╕=в:
3в0
*К'
inputs         ┤┤
p
к "+К(
unknown         ┤┤Ю
.__inference_random_rotation_layer_call_fn_9706l=в:
3в0
*К'
inputs         ┤┤
p 
к "+К(
unknown         ┤┤─
E__inference_random_zoom_layer_call_and_return_conditional_losses_9942{╣=в:
3в0
*К'
inputs         ┤┤
p
к "6в3
,К)
tensor_0         ┤┤
Ъ └
E__inference_random_zoom_layer_call_and_return_conditional_losses_9946w=в:
3в0
*К'
inputs         ┤┤
p 
к "6в3
,К)
tensor_0         ┤┤
Ъ Ю
*__inference_random_zoom_layer_call_fn_9835p╣=в:
3в0
*К'
inputs         ┤┤
p
к "+К(
unknown         ┤┤Ъ
*__inference_random_zoom_layer_call_fn_9840l=в:
3в0
*К'
inputs         ┤┤
p 
к "+К(
unknown         ┤┤╘
D__inference_sequential_layer_call_and_return_conditional_losses_7971Л
*+89FG`ahiOвL
EвB
8К5
module_wrapper_input         ┤┤
p 

 
к ",в)
"К
tensor_0         
Ъ ┌
D__inference_sequential_layer_call_and_return_conditional_losses_8013С╖╕╣*+89FG`ahiOвL
EвB
8К5
module_wrapper_input         ┤┤
p

 
к ",в)
"К
tensor_0         
Ъ ┼
D__inference_sequential_layer_call_and_return_conditional_losses_8150}
*+89FG`ahiAв>
7в4
*К'
inputs         ┤┤
p 

 
к ",в)
"К
tensor_0         
Ъ ╠
D__inference_sequential_layer_call_and_return_conditional_losses_8471Г╖╕╣*+89FG`ahiAв>
7в4
*К'
inputs         ┤┤
p

 
к ",в)
"К
tensor_0         
Ъ ╫
D__inference_sequential_layer_call_and_return_conditional_losses_9056О╖╕╣LвI
Bв?
5К2
random_flip_input         ┤┤
p

 
к "6в3
,К)
tensor_0         ┤┤
Ъ ╧
D__inference_sequential_layer_call_and_return_conditional_losses_9075ЖLвI
Bв?
5К2
random_flip_input         ┤┤
p 

 
к "6в3
,К)
tensor_0         ┤┤
Ъ ╠
D__inference_sequential_layer_call_and_return_conditional_losses_9616Г╖╕╣Aв>
7в4
*К'
inputs         ┤┤
p

 
к "6в3
,К)
tensor_0         ┤┤
Ъ ├
D__inference_sequential_layer_call_and_return_conditional_losses_9620{Aв>
7в4
*К'
inputs         ┤┤
p 

 
к "6в3
,К)
tensor_0         ┤┤
Ъ о
)__inference_sequential_layer_call_fn_7421А
*+89FG`ahiOвL
EвB
8К5
module_wrapper_input         ┤┤
p 

 
к "!К
unknown         ┤
)__inference_sequential_layer_call_fn_7935Ж╖╕╣*+89FG`ahiOвL
EвB
8К5
module_wrapper_input         ┤┤
p

 
к "!К
unknown         Я
)__inference_sequential_layer_call_fn_8071r
*+89FG`ahiAв>
7в4
*К'
inputs         ┤┤
p 

 
к "!К
unknown         е
)__inference_sequential_layer_call_fn_8102x╖╕╣*+89FG`ahiAв>
7в4
*К'
inputs         ┤┤
p

 
к "!К
unknown         ▒
)__inference_sequential_layer_call_fn_9100Г╖╕╣LвI
Bв?
5К2
random_flip_input         ┤┤
p

 
к "+К(
unknown         ┤┤и
)__inference_sequential_layer_call_fn_9112{LвI
Bв?
5К2
random_flip_input         ┤┤
p 

 
к "+К(
unknown         ┤┤е
)__inference_sequential_layer_call_fn_9341x╖╕╣Aв>
7в4
*К'
inputs         ┤┤
p

 
к "+К(
unknown         ┤┤Э
)__inference_sequential_layer_call_fn_9346pAв>
7в4
*К'
inputs         ┤┤
p 

 
к "+К(
unknown         ┤┤╟
"__inference_signature_wrapper_8046а
*+89FG`ahi_в\
в 
UкR
P
module_wrapper_input8К5
module_wrapper_input         ┤┤"1к.
,
outputs!К
outputs         