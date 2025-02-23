import numpy as np
import paddle
import paddle_quantum as pq
from paddle_quantum.ansatz.circuit import Circuit

# 定义量子比特数
num_qubits = 2

# 初始化量子电路
cir = Circuit(num_qubits)

# 添加量子门
theta = paddle.to_tensor([np.pi / 2, np.pi / 4])
cir.ry(theta[0], 0)
cir.rz(theta[1], 1)

# 运行电路
output_state = cir(pq.state.zero_state(num_qubits))

# 测量结果
measurement_result = cir.measure(shots=1024)
print("Measurement result:", measurement_result)