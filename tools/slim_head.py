"""Rebuilds the end of a recognizer model so that it returns only the columns the engine asks for.

Run: .venv/Scripts/python tools/slim_head.py SOURCE.onnx DEST.onnx (fetch_libs.py calls it for the add-on).

The original head is MatMul + Add + Softmax over the whole dictionary: in the Latin mode 18.7k
probabilities per CTC step, though the engine keeps only ~760 of them. On full-width lines this output
(and the logits before it) takes hundreds of MB. The new head returns exactly the same probabilities
for the asked columns: exp(logit - logsumexp of all logits). Both are computed in slices
of the dictionary, so the full-width tensor never exists. The asked columns come in the extra inputs
keep0, keep1, ...: one per slice, numbered from the slice start (the engine splits them).
Needs the onnx package (only here, not in the add-on).
"""

import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# Dictionary columns per slice: measured, 1024-2048 is the fastest; memory is the same.
SLICE = 2048


def slim_head(src, dst):
	model = onnx.load(str(src))
	graph = model.graph
	inits = {i.name: i for i in graph.initializer}
	producer = {out: node for node in graph.node for out in node.output}
	out = graph.output[0].name
	softmax = producer[out]
	head = [softmax]
	node = producer[softmax.input[0]]
	while node.op_type == "Identity":
		head.append(node)
		node = producer[node.input[0]]
	add = node
	matmul = producer[add.input[0]]
	if softmax.op_type != "Softmax" or add.op_type != "Add" or matmul.op_type != "MatMul":
		raise SystemExit(f"Ошибка: у {src} нет головы MatMul + Add + Softmax, переделать нельзя.")
	features, weights = matmul.input
	bias = add.input[1]
	w = numpy_helper.to_array(inits[weights])
	b = numpy_helper.to_array(inits[bias])
	classes = w.shape[1]
	for node in [matmul, add, *head]:
		graph.node.remove(node)
	# The weights go into the slices below: each is kept once (onnxruntime packs them for MatMul).
	graph.initializer.remove(inits[weights])
	graph.initializer.remove(inits[bias])
	# Old models may also list initializers among the inputs.
	for inp in [inp for inp in graph.input if inp.name in (weights, bias)]:
		graph.input.remove(inp)

	def const(name, value):
		graph.initializer.append(numpy_helper.from_array(np.asarray(value), f"slim_{name}"))
		return f"slim_{name}"

	nodes = []
	lses = []
	picked = []
	for i, start in enumerate(range(0, classes, SLICE)):
		end = min(classes, start + SLICE)
		n = f"slim_{i}_"
		nodes += [
			helper.make_node("MatMul", [features, const(f"w{i}", w[:, start:end])], [n + "m"]),
			helper.make_node("Add", [n + "m", const(f"b{i}", b[start:end])], [n + "logits"]),
			# Log-sum-exp of the slice with the maximum subtracted first (exp must not overflow).
			# Measured: these plain nodes are faster than ReduceLogSumExp of onnxruntime.
			helper.make_node("ReduceMax", [n + "logits"], [n + "max"], axes=[-1], keepdims=1),
			helper.make_node("Sub", [n + "logits", n + "max"], [n + "shifted"]),
			helper.make_node("Exp", [n + "shifted"], [n + "exp"]),
			helper.make_node("ReduceSum", [n + "exp"], [n + "sum"], axes=[-1], keepdims=1),
			helper.make_node("Log", [n + "sum"], [n + "log"]),
			helper.make_node("Add", [n + "log", n + "max"], [n + "lse"]),
			# The asked columns of this slice, numbered from its start (input keep<i>, may be empty).
			helper.make_node("Gather", [n + "logits", f"keep{i}"], [n + "picked"], axis=-1),
		]
		graph.input.append(helper.make_tensor_value_info(f"keep{i}", TensorProto.INT64, [f"keep{i}"]))
		lses.append(n + "lse")
		picked.append(n + "picked")
	# The slices go in order, so the columns come out sorted.
	nodes += [
		helper.make_node("Concat", lses, ["slim_lses"], axis=-1),
		helper.make_node("ReduceLogSumExp", ["slim_lses"], ["slim_lse"], axes=[-1], keepdims=1),
		helper.make_node("Concat", picked, ["slim_keepLogits"], axis=-1),
		helper.make_node("Sub", ["slim_keepLogits", "slim_lse"], ["slim_keepLog"]),
		helper.make_node("Exp", ["slim_keepLog"], [out]),
	]
	graph.node.extend(nodes)
	dims = graph.output[0].type.tensor_type.shape.dim
	dims[-1].Clear()
	dims[-1].dim_param = "keep"
	# The output no longer tells the dictionary size (the engine checks it against inference.yml),
	# and the engine needs the slice size to split its columns among the inputs.
	for key, value in (("classes", classes), ("slice", SLICE)):
		prop = model.metadata_props.add()
		prop.key, prop.value = key, str(value)
	onnx.checker.check_model(model)
	onnx.save(model, str(dst))


def main():
	if len(sys.argv) != 3:
		print(__doc__)
		return 2
	slim_head(sys.argv[1], sys.argv[2])
	print(f"Готово: {sys.argv[2]}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
