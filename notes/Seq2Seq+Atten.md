Seq2Seq:

At any time step $t'$ at decoder stage, the RNN takes the output $y'_{t-1}$ from the previous time step and the context variable $c$ (Encoder's output) as input.

![image-20211015213558901](Seq2Seq+Atten.assets/image-20211015213558901.png)