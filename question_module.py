import ipywidgets as widgets
from IPython.display import display

def question_1():
    question = widgets.HTML(value="With a sampling frequency of 20 Hz, we can recover the number of cycles in our original signal. </br>"
                                "<b>What is the minimum sampling frequency required to also recover the signal amplitude?</b>")
    answer = widgets.Dropdown(
        options=['21 Hz', '41 Hz', '200 Hz', '400 Hz'],
        description='Choose:'
    )
    button = widgets.Button(description="Check Answer")
    output = widgets.Output()

    def check_answer(b):
        with output:
            output.clear_output()
            if answer.value == '41 Hz':
                print("✅ Correct!")
            else:
                print("❌ Try again.")

    button.on_click(check_answer)
    display(question, answer, button, output)

def question_2():
    question = widgets.HTML(value="Slow wave sleep produces signals between 0.1 and 4 Hz. </br>"
                                "<b>Is a sampling frequency of 100 Hz sufficient to reproduce the signal?</b>")
    answer = widgets.Dropdown(
        options=['Yes', 'No'],
        description='Choose:'
    )
    button = widgets.Button(description="Check Answer")
    output = widgets.Output()

    def check_answer(b):
        with output:
            output.clear_output()
            if answer.value == 'Yes':
                print("✅ Correct! 100 Hz satisfies the Nyquist Criterion. "
                      "This sampling frequency is fine for most EEG data, but wouldn't be enough for high-frequency spiking data, "
                      "which is typically sampled at 40 kHz.")
            else:
                print("❌ Try again.")

    button.on_click(check_answer)
    display(question, answer, button, output)

def question_3():
    question = widgets.HTML(value="<b>Which edges are accentuated by the <i>Vertical</i> filter?</b>")
    answer = widgets.Dropdown(
        options=['All edges', 'Vertical edges', 'Horizontal edges', 'No edges'],
        description='Choose:'
    )
    button = widgets.Button(description="Check Answer")
    output = widgets.Output()

    def check_answer(b):
        with output:
            output.clear_output()
            if answer.value == 'Horizontal edges':
                print("✅ Correct!")
            else:
                print("❌ Try again.")

    button.on_click(check_answer)
    display(question, answer, button, output)
