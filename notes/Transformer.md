Attention Mechanisms

Query和Keys经过Attention scoring function算出注意力分数，归一化之后得到Attention Weights。注意力权重和Values相乘得到输出。

![image-20211013210912202](Transformer.assets/image-20211013210912202.png)

Suppose that we have a query $\mathbf{q} \in \mathbb{R}^q$ and $m$ key-value pairs $(\mathbf{k}_1, \mathbf{v}_1), \ldots, (\mathbf{k}_m, \mathbf{v}_m)$(k1,v1),…,(km,vm), where any $\mathbf{k}_i \in \mathbb{R}^k$ and any $\mathbf{v}_i \in \mathbb{R}^v$. The attention pooling is instantiated as a weighted sum of the *values*:
$$
f(\mathbf{q}, (\mathbf{k}_1, \mathbf{v}_1), \ldots, (\mathbf{k}_m, \mathbf{v}_m)) = \sum_{i=1}^m \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i \in \mathbb{R}^v,
$$
where the attention weight(scalar) $\alpha(\mathbf{q}, \mathbf{k}_i)$ for $\mathbf{v}_i$ is computed over $\mathbf{q}, \mathbf{k}_i$ by the softmax operation of an attention scoring function $a$ that maps **two vectors** to a **scalar**:
$$
\alpha(\mathbf{q}, \mathbf{k}_i) = \mathrm{softmax}(a(\mathbf{q}, \mathbf{k}_i)) = \frac{\exp(a(\mathbf{q}, \mathbf{k}_i))}{\sum_{j=1}^m \exp(a(\mathbf{q}, \mathbf{k}_j))} \in \mathbb{R}.
$$

## Scaled Dot-Product Attention

This kind of attention requires both the *query* and the *key* have the same vector length $d$. The scaled dot-product attention scoring function is:
$$
a(\mathbf{q}, \mathbf{k}) = \mathbf{q}^\top \mathbf{k} / \sqrt{d}
$$
In practice, for general situations, we need to compute attention for $n$ queries and $m$ key-value pairs, where queries and keys are of length $d$ and values are of length $v$. The scaled dot-product attention of queries $\mathbf Q\in\mathbb R^{n\times d}$, keys $\mathbf K \in \mathbb R ^ {m \times d}$, and values $\mathbf V\in\mathbb R^{m\times v}$ is:
$$
\mathrm{softmax}\left(\frac{\mathbf Q \mathbf K^\top }{\sqrt{d}}\right) \mathbf V \in \mathbb{R}^{n\times v}.
$$
