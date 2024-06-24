import gradio as gr
import random
from recurrentgpt import RecurrentGPT
from human_simulator import Human
from sentence_transformers import SentenceTransformer
from utils import get_init, parse_instructions
from starlette.requests import Request
import re


_CACHE = {}


# Build the semantic search model
embedder = SentenceTransformer('multi-qa-mpnet-base-cos-v1')

def init_prompt(novel_type, description, language):
    if description == "":
        description = ""
    else:
        description = " about " + description
    if language != "":
        description += f" in {language} language"
    return f"""
Please write a {novel_type} novel{description}with 50 chapters. Follow the format below precisely:

Begin with the name of the novel.
Next, write an outline for the first chapter. The outline should describe the background and the beginning of the novel.
Write the first three paragraphs with their indication of the novel based on your outline. Write in a novelistic style and take your time to set the scene.
Write a summary that captures the key information of the three paragraphs.
Finally, write three different instructions for what to write next, each containing around five sentences. Each instruction should present a possible, interesting continuation of the story.
The output format should follow these guidelines:
Name: <name of the novel>
Outline: <outline for the first chapter>
Paragraph 1: <content for paragraph 1>
Paragraph 2: <content for paragraph 2>
Paragraph 3: <content for paragraph 3>
Summary: <content of summary>
Instruction 1: <content for instruction 1>
Instruction 2: <content for instruction 2>
Instruction 3: <content for instruction 3>

Make sure to be precise and follow the output format strictly.

"""

def init(novel_type, description, language, save_story, request: gr.Request):
    out_file = None
    if save_story == "Yes":
        out_file = f"{novel_type}_{description}_{language}.txt"

    cookie = request.headers.get('cookie', None)
    if cookie is None:
        cookie = ""
    else:
        cookie = request.headers['cookie']
        cookie = cookie.split('; _gat_gtag')[0]

    if novel_type == "":
        novel_type = "Science Fiction"
    global _CACHE

    init_paragraphs = get_init(text=init_prompt(novel_type, description, language),response_file=out_file)
    
    start_input_to_human = {
        'output_paragraph': init_paragraphs['Paragraph 3'],
        'input_paragraph': '\n\n'.join([init_paragraphs['Paragraph 1'], init_paragraphs['Paragraph 2']]),
        'output_memory': init_paragraphs['Summary'],
        "output_instruction": [init_paragraphs['Instruction 1'], init_paragraphs['Instruction 2'], init_paragraphs['Instruction 3']]
    }

    _CACHE[cookie] = {"start_input_to_human": start_input_to_human,
                      "init_paragraphs": init_paragraphs}
    written_paras = f"""Title: {init_paragraphs['name']}

Outline: {init_paragraphs['Outline']}

Paragraphs:

{start_input_to_human['input_paragraph']}"""
    long_memory = parse_instructions([init_paragraphs['Paragraph 1'], init_paragraphs['Paragraph 2']])
    # short memory, long memory, current written paragraphs, 3 next instructions
    print("inital", written_paras)
    return start_input_to_human['output_memory'], long_memory, written_paras, init_paragraphs['Instruction 1'], init_paragraphs['Instruction 2'], init_paragraphs['Instruction 3']

def step(novel_type, description, language, short_memory, long_memory, save_story, instruction1, instruction2, instruction3, current_paras, request: gr.Request,):
    out_file = None
    if save_story == "Yes":
        out_file = f"{novel_type}_{description}_{language}.txt"
    
    if current_paras == "":
        return "", "", "", "", "", ""
    global _CACHE
    print("control step request", request)
    cookie = request.headers.get('cookie', None)
    if cookie is None:
        # Handle the case where the cookie is not present
        cookie = ""
    else:
        cookie = request.headers['cookie']
        cookie = cookie.split('; _gat_gtag')[0]
    cache = _CACHE[cookie]

    if "writer" not in cache:
        start_input_to_human = cache["start_input_to_human"]
        start_input_to_human['output_instruction'] = [
            instruction1, instruction2, instruction3]
        init_paragraphs = cache["init_paragraphs"]
        human = Human(input=start_input_to_human,
                      memory=None, embedder=embedder, language=language, output_file=out_file)
        human.step()
        start_short_memory = init_paragraphs['Summary']
        writer_start_input = human.output

        # Init writerGPT
        writer = RecurrentGPT(input=writer_start_input, short_memory=start_short_memory, long_memory=[
            init_paragraphs['Paragraph 1'], init_paragraphs['Paragraph 2']], memory_index=None, embedder=embedder, language=language, output_file=out_file)
        cache["writer"] = writer
        cache["human"] = human
        writer.step()
    else:
        human = cache["human"]
        writer = cache["writer"]
        output = writer.output
        output['output_memory'] = short_memory
        #randomly select one instruction out of three
        instruction_index = random.randint(0,2)
        output['output_instruction'] = [instruction1, instruction2, instruction3][instruction_index]
        human.input = output
        human.step()
        writer.input = human.output
        writer.step()

    long_memory = parse_instructions(writer.long_memory)
    # short memory, long memory, current written paragraphs, 3 next instructions
    return writer.output['output_memory'], long_memory, current_paras + '\n\n' + writer.output['input_paragraph'], human.output['output_instruction'], *writer.output['output_instruction']


def controled_step(novel_type, description, language, short_memory, save_story, long_memory, selected_instruction, current_paras, request: gr.Request, ):
    out_file = None
    if save_story == "Yes":
        out_file = f"{novel_type}_{description}_{language}.txt"
    
    if current_paras == "":
        return "", "", "", "", "", ""
    global _CACHE
    print("control step request", request)
    cookie = request.headers.get('cookie', None)
    if cookie is None:
        # Handle the case where the cookie is not present
        cookie = ""
    else:
        cookie = request.headers['cookie']
        cookie = cookie.split('; _gat_gtag')[0]
    cache = _CACHE[cookie]

    if "writer" not in cache:
        start_input_to_human = cache["start_input_to_human"]
        start_input_to_human['output_instruction'] = selected_instruction
        init_paragraphs = cache["init_paragraphs"]
        human = Human(input=start_input_to_human,
                      memory=None, embedder=embedder, language=language, output_file=out_file)
        human.step()
        start_short_memory = init_paragraphs['Summary']
        writer_start_input = human.output

        # Init writerGPT
        writer = RecurrentGPT(input=writer_start_input, short_memory=start_short_memory, long_memory=[
            init_paragraphs['Paragraph 1'], init_paragraphs['Paragraph 2']], memory_index=None, embedder=embedder, language=language, output_file=out_file)
        cache["writer"] = writer
        cache["human"] = human
        writer.step()
    else:
        human = cache["human"]
        writer = cache["writer"]
        output = writer.output
        output['output_memory'] = short_memory
        output['output_instruction'] = selected_instruction
        human.input = output
        human.step()
        writer.input = human.output
        writer.step()
    # long_memory = parse_instructions(writer.long_memory)
    long_memory = parse_instructions(writer.long_memory)
    # short memory, long memory, current written paragraphs, 3 next instructions
    return writer.output['output_memory'], long_memory, current_paras + '\n\n' + writer.output['input_paragraph'], selected_instruction, *writer.output['output_instruction']


# SelectData is a subclass of EventData
def on_select(instruction1, instruction2, instruction3, evt: gr.SelectData):
    selected_plan = int(evt.value.replace("Instruction ", ""))
    selected_plan = [instruction1, instruction2, instruction3][selected_plan-1]
    return selected_plan

with gr.Blocks(title="RecurrentGPT", css="footer {visibility: hidden}", theme="default") as demo:
    gr.Markdown(
        """
    # Auto-Generation & Select-Generation
    # Novel Type is REQUIRED
    # Topic is not required but good to have.
    # Default language is English. If you want the story to be other language, please specify.
    # Must Initial Novel FIRST.
    """)

    with gr.Tab("Auto-Generation"):
        with gr.Column():
            with gr.Row():
                novel_type = gr.Textbox(
                    label="Novel Type", placeholder="e.g. science fiction")
                description = gr.Textbox(label="Topic")
                language = gr.Textbox(label="Language")
                save_story = gr.Radio(choices=["Yes", "No"], label="Save Story")
            gr.Examples(["Science Fiction", "Romance", "Mystery", "Fantasy",
                        "Historical", "Horror", "Thriller", "Western", "Young Adult"],
                        inputs=[novel_type], elem_id="example_selector")
            btn_init = gr.Button(
                "Init Novel Generation", elem_id="init_button")
            written_paras = gr.Textbox(
                label="Written Paragraphs (editable)", lines=21)

        with gr.Column():
            gr.Markdown("### Memory Module")
            short_memory = gr.Textbox(
                label="Short-Term Memory (editable)", lines=3)
            long_memory = gr.Textbox(
                label="Long-Term Memory (editable)", lines=6)
            gr.Markdown("### Instruction Module")
            instruction1 = gr.Textbox(
                label="Instruction 1 (editable)", lines=4)
            instruction2 = gr.Textbox(
                label="Instruction 2 (editable)", lines=4)
            instruction3 = gr.Textbox(
                label="Instruction 3 (editable)", lines=4)
            selected_plan = gr.Textbox(
                label="Revised Instruction (from last step)", lines=2)
        btn_step = gr.Button("Next Step", elem_id="step_button")
        btn_init.click(init, inputs=[novel_type, description, language, save_story], outputs=[
            short_memory, long_memory, written_paras, instruction1, instruction2, instruction3])
        btn_step.click(step, inputs=[novel_type, description, language, save_story, short_memory, long_memory, instruction1, instruction2, instruction3, written_paras], outputs=[
            short_memory, long_memory, written_paras, selected_plan, instruction1, instruction2, instruction3])
    
    # new tab
    with gr.Tab("Select-Generation"):
        with gr.Column():
            with gr.Row():
                novel_type = gr.Textbox(
                    label="Novel Type", placeholder="e.g. science fiction")
                description = gr.Textbox(label="Topic")
                language = gr.Textbox(label="Language")
                save_story = gr.Radio(choices=["Yes", "No"], label="Save Story")
            gr.Examples(["Science Fiction", "Romance", "Mystery", "Fantasy",
                        "Historical", "Horror", "Thriller", "Western", "Young Adult"],
                        inputs=[novel_type], elem_id="example_selector")
            btn_init = gr.Button(
                "Init Novel Generation", elem_id="init_button")
            written_paras = gr.Textbox(
                label="Written Paragraphs (editable)", lines=21)

        with gr.Column():
            gr.Markdown("### Memory Module")
            short_memory = gr.Textbox(
                label="Short-Term Memory (editable)", lines=3)
            long_memory = gr.Textbox(
                label="Long-Term Memory (editable)", lines=6)
            gr.Markdown("### Instruction Module")
            instruction1 = gr.Textbox(
                label="Instruction 1 (editable)", lines=4)
            instruction2 = gr.Textbox(
                label="Instruction 2 (editable)", lines=4)
            instruction3 = gr.Textbox(
                label="Instruction 3 (editable)", lines=4)
            last_step = gr.Textbox(
                label="Instruction from last step", lines=2)
        with gr.Column():
            with gr.Column(scale=1, min_width=100):
                            selected_plan = gr.Radio(["Instruction 1", "Instruction 2", "Instruction 3"], label="selected_plan")
                                                    #  info="Select the instruction you want to revise and use for the next step generation.")
            with gr.Column(scale=3, min_width=300):
                            selected_instruction = gr.Textbox(
                                label="Selected Instruction (editable)", max_lines=5, lines=5)

        btn_step = gr.Button("Next Step", elem_id="step_button")
        btn_init.click(init, inputs=[novel_type, description, language, save_story], outputs=[
            short_memory, long_memory, written_paras, instruction1, instruction2, instruction3])
        btn_step.click(controled_step, inputs=[novel_type, description, language, save_story, short_memory, long_memory, selected_instruction, written_paras], outputs=[
            short_memory, long_memory, written_paras, last_step, instruction1, instruction2, instruction3])
        selected_plan.select(on_select, inputs=[
                             instruction1, instruction2, instruction3], outputs=[selected_instruction])

    demo.queue(max_size=20)
    demo.launch(max_threads=1, inbrowser=True, share=True)
    #demo.queue(concurrency_count=1)

if __name__ == "__main__":
    demo.launch(server_port=8005, share=True,
                server_name="0.0.0.0", show_api=False)