import gradio as gr
import numpy as np
import pydicom
from predict import predict_denoised_slice


def denoise(dicom_file, Window_Width=1600, Window_Level=-200):
    print(dicom_file[0], Window_Width, Window_Level)
    dcm = pydicom.read_file(dicom_file[0], force=True)
    ct_data = np.array(dcm.pixel_array, "float32") - 1000
    ct_data = np.clip((ct_data - (Window_Level - Window_Width/ 2)) / Window_Width, 0, 1)
    denoised = np.clip(predict_denoised_slice(ct_data), 0, 1)
    return ct_data, denoised


# with gr.Blocks() as demo:
#     gr.Markdown("""<h1 align="center">
#     HorusEye for CT denoising.
#     </h1>""")
#     gr.Markdown(
#         """<h3>
#         A temp GUI for HorusEye Feel free to upload a .dcm file with 512×512 size and see the denoised results.
#         </h3>""")
#
#     inp = [gr.Files(type="filepath", label="Upload DICOM File"),
#            "number", "number"]
#     out = ["image", "image"]
#     inp.change(denoise, inp, out)


demo = gr.Interface(
    # gr.Markdown("""<h1 align="center"> HorusEye for CT denoising.</h1>"""),
    # gr.Markdown(
    #     """<h3>
    #     A temp GUI for HorusEye Feel free to upload a .dcm file with 512×512 size and see the denoised results.
    #     </h3>"""),
    denoise,
    inputs=[gr.Files(type="filepath", label="Upload DICOM File"),
            gr.Number(label="Window Width"), gr.Number(label="Window Level")],
    outputs=[gr.Image(label="Raw CT slice"), gr.Image(label="Denoised results by HorusEye")],
    examples=[[["./chest_example.dcm"], 1600, -200],
              [["./abdomen_example.dcm"], 400, 40]],
    title="HorusEye for CT denoising",
    description="A temp GUI for HorusEye Feel free to upload a .dcm file with 512×512 size, enter the window width and window level, and see the denoised results.",
    article="Note: The CT slice is first clipped according to the given window width and window level. "
            "A too small window width will change the noise patterns and could adversely affect the denoising performance."
            "You can try the following window width and window level: [-1000, 600] for the lung window, [-200, 300] for the mediastinal window, [-160, 240] for the abdomen window.")
demo.launch()

