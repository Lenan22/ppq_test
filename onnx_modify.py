import torch
from torch.autograd import Function
from collections import OrderedDict
import onnx_graphsurgeon as gs
import onnx
import numpy as np






#ClipPlugin for relu6 has been a built-in op in tensorrt, and we map the relu6 to CustomClipPlugin only for demo.
def replace_top_with_customTopkPlugin(old_onnx_filename, new_onnx_filename):  
    model = onnx.load(old_onnx_filename)

    graph = model.graph
    # graph.cleanup()
    
    #map Clip op to CustomClipPlugin
    for i in range(len(graph.node)):
        node = graph.node[i]

        output = node.output

        if '594' in output:
            node.name = "reshze_" + str(i)
            print("---")

        if '595' in output:
            node.name = "reshze_" + str(i)
            print("---")

        # if node.name in topk_constant_1000:
        #     output_tensor_name = str(fname) + "_topk"
        #     node.inputs[1] = gs.ir.tensor.Constant(output_tensor_name,np.array([1000]))

        # if node.name in topk_constant_624:
        #     output_tensor_name = str(fname) + "_topk"
        #     node.inputs[1] = gs.ir.tensor.Constant(output_tensor_name,np.array([624]))

        # if node.name in topk_constant_100:
        #     output_tensor_name = str(fname) + "_topk"
        #     node.inputs[1] = gs.ir.tensor.Constant(output_tensor_name,np.array([100]))

        # if "Clip" in node.name :
        #     node.name = "Clip_plugin"
        #     node.op = "CustomClipPlugin" # keep the same with CLIP_PLUGIN_NAME in customClipPlugin.cpp
        #     node.attrs = OrderedDict({"clipMin":0.0, "clipMax":6.0})

    # onnx.checker.check_model(model)
    onnx.save(model, new_onnx_filename)    


# OrderedDict([('axis', 1), ('largest', 1), ('sorted', 1)])


if __name__ == '__main__':
    onnx_filename = "Retinanet_3_450_620.onnx"
    # onnx_filename = "models/new_end2end_float_static.onnx"
    # export_onnx(onnx_filename)
    # print_onnx_model(onnx_filename)

    new_onnx_filename = "Retinanet_3_450_620_add_resize_name.onnx"
    # new_onnx_filename = "models/new_end2end_float_static.onnx"

    replace_top_with_customTopkPlugin(onnx_filename, new_onnx_filename)


