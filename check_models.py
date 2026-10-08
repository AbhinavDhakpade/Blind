import hashlib
import onnxruntime as ort

files = [
    "models/object_detection/yolov8n.onnx",
    "models/face_recognition/mobilefacenet.onnx",
    "models/face_recognition/w600k_mbf.onnx",
]

for f in files:
    sess = ort.InferenceSession(f, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    md5 = hashlib.md5(open(f, "rb").read()).hexdigest()
    print(f)
    print("  input :", inp.name, inp.shape)
    print("  output:", out.name, out.shape)
    print("  md5   :", md5)