from gpt4all import GPT4All

model_path = r"E:\AIML_Project\xray_disease_detector_gui2.zip\xray_disease_detector_gui2\models\gpt4all-falcon.Q4_0.gguf"
model = GPT4All(model_path)  # Load the model from local path

def medical_chatbot():
    print("🤖 AI Medical Chatbot - Type 'exit' to stop")
    while True:
        user_input = input("🧝‍♀️ You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye! Stay healthy! 😊")
            break
        response = model.generate(user_input, max_tokens=200)
        print(f"Chatbot: {response}")

medical_chatbot()
 