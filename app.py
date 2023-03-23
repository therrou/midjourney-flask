import flask
import replicate
import os
os.environ["REPLICATE_API_TOKEN"] = "753c3784b5052ae42b5f0de056386860ab943540"

app = flask.Flask(__name__)

model = replicate.models.get("tstramer/midjourney-diffusion")
version = model.versions.get("436b051ebd8f68d23e83d22de5e198e0995357afef113768c20f0b6fcef23c8b")

@app.route('/', methods=['GET', 'POST'])
def index():
    if flask.request.method == 'POST':
        # Get user input from the form
        prompt = flask.request.form['prompt']
        negative_prompt = flask.request.form['negative_prompt']
        num_outputs = int(flask.request.form['num_outputs'])

        # Set the other input parameters to their default values
        width = 768
        height = 768
        prompt_strength = 0.8
        num_inference_steps = 50
        guidance_scale = 7.5
        scheduler = "DPMSolverMultistep"

        # Call the model with the user input
        inputs = {
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'width': width,
            'height': height,
            'prompt_strength': prompt_strength,
            'num_outputs': num_outputs,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'scheduler': scheduler,
            # 'seed': seed,
        }
        output = version.predict(**inputs)
        # show on the console the output
        # Render the output image(s) on the web page
        return flask.render_template('index.html', output=output[0])

    else:
        # Show the input form to the user
        return flask.render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
