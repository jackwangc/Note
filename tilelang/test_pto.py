import ctypes
import torch
import torch.nn.functional as F

def ref_chunk_h(k, w, u, g, C):
	B, H, L, DK = k.shape
	DV = u.shape[-1]
	chunk_num = (L + C - 1) // C
	s = torch.zeros((B, H, chunk_num, DK, DV)).npu().to(torch.float)
	new_v = torch.zeros((B, H, L, DV)).npu().to(torch.float)
	k = k.float()
	u = u.float()

	for i in range(chunk_num):
		las_s = s[:, :, i, :, :]
		k_c = k[:, :, i * C : (i + 1) * C, :]
		w_c = w[:, :, i * C : (i + 1) * C, :]
		u_c = u[:, :, i * C : (i + 1) * C, :]
		g_c = g[:, :, i * C : (i + 1) * C]
		ws = torch.matmul(w_c, las_s.to(torch.float16)).float()
		new_v_c = u_c - ws
		new_v[:, :, i * C : (i + 1) * C, :] = new_v_c
		g_last = g[:, :, (i + 1) * C - 1].view(B, H, 1, 1)
		coeff_k = g_last - g_c.view(B, H, C, 1)
		g_last = torch.exp(g_last)
		coeff_k = torch.exp(coeff_k)
		k_c = (k_c * coeff_k).transpose(-2, -1)
		las_s = las_s * g_last
		kv = torch.matmul(k_c.to(torch.float16), new_v_c.to(torch.float16)).float()
		s_c = las_s + kv
		if i < chunk_num - 1:
			s[:, :, i + 1, :, :] = s_c

	return s.to(torch.float16), new_v.to(torch.float16), s_c.to(torch.float16)

def ref_chunk_cumsum(g, C):
	B, H, L = g.shape
	chunk_num = (L + C - 1) // C
	g = g.view(B, H, chunk_num, C)
	g_sum = torch.cumsum(g, dim = -1)
	g_sum = g_sum.view(B, H, L)
	return g_sum

# Load library (use pto.so for PTO kernel, ascendc.so for AscendC kernel)
lib_path = "./pto.so"
lib_path = "./ascendc.so"
lib = ctypes.CDLL(lib_path)

# Get NPU stream
stream = torch.npu.current_stream()._as_parameter_

# Test configuration (same as opt_gdn_chunk_h.py)
torch.manual_seed(0)
torch.set_printoptions(threshold = float('inf'), sci_mode = True)

test_configs = [
	(2, 16, 16384, 128, 128, 128),
]

for B, H, L, DK, DV, C in test_configs:
	print(f"Testing Hidden State with B={B}, H={H}, L={L}, DK={DK}, DV={DV}, C={C}")

	BV = DV
	chunk_num = (L + C - 1) // C  # = 128
	bv_num = (DV + DV - 1) // DV  # = 1

	# Initialize input tensors (same as opt_gdn_chunk_h.py)
	k = torch.randn((B, H, L, DK)).npu().to(torch.float16)
	w = torch.randn((B, H, L, DK)).npu().to(torch.float16)
	u = torch.randn((B, H, L, DV)).npu().to(torch.float16)
	g = torch.randn((B, H, L)).npu().to(torch.float)
	g = F.logsigmoid(g)
	k, w = F.normalize(k, dim=-1, p=2), F.normalize(w, dim=-1, p=2)

	# Compute g chunk cumsum (reference)
	g = ref_chunk_cumsum(g, C)

	# Workspace tensors (matching the kernel function signatures)
	workspace_1 = torch.zeros((B * H * bv_num, C, BV)).npu().to(torch.float16)  # [32, 128, 128]
	workspace_2 = torch.zeros((B * H * bv_num, C, DK)).npu().to(torch.float16)  # [32, 128, 128]
	workspace_3 = torch.zeros((B * H * bv_num, DK, BV)).npu().to(torch.float16)  # [32, 128, 128] - need to be 0
	workspace_4 = torch.zeros((B * H * bv_num, DK, BV)).npu().to(torch.float16)  # [32, 128, 128]

	# Output tensors
	S = torch.zeros((B, H, chunk_num, DK, DV)).npu().to(torch.float16)  # [2, 16, 128, 128, 128] - need to be 0
	V = torch.zeros((B, H, L, DV)).npu().to(torch.float16)  # [2, 16, 16384, 128]
	FS = torch.zeros((B, H, DK, DV)).npu().to(torch.float16)  # [2, 16, 128, 128]

	def call_kernel():
		return lib.call(
			ctypes.c_void_p(k.data_ptr()),
			ctypes.c_void_p(w.data_ptr()),
			ctypes.c_void_p(u.data_ptr()),
			ctypes.c_void_p(g.data_ptr()),
			ctypes.c_void_p(workspace_1.data_ptr()),
			ctypes.c_void_p(workspace_2.data_ptr()),
			ctypes.c_void_p(workspace_3.data_ptr()),
			ctypes.c_void_p(workspace_4.data_ptr()),
			ctypes.c_void_p(S.data_ptr()),
			ctypes.c_void_p(V.data_ptr()),
			ctypes.c_void_p(FS.data_ptr()),
			stream
		)

	# Call the kernel
	call_kernel()

	# Compute reference values
	ref_s, ref_new_v, ref_final_s = ref_chunk_h(k, w, u, g, C)

	# Verify precision
	torch.testing.assert_close(S.cpu(), ref_s.cpu(), rtol=1e-5, atol=1e-5)
	torch.testing.assert_close(V.cpu(), ref_new_v.cpu(), rtol=1e-5, atol=1e-5)
	torch.testing.assert_close(FS.cpu(), ref_final_s.cpu(), rtol=1e-5, atol=1e-5)

	print("Test passed!")

print("Kernel Output Match!")
