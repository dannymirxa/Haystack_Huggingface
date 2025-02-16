from generate_simple_response import generate_simple_response


if __name__=="__main__":
    while True:
        prompt = input("You: ")
        if prompt.lower() in ["quit", "exit", "goodbye"]:
            break

        response = str(generate_simple_response(prompt)["replies"][0]).split("text='")[1].split("')],",1)[0]

        print("Chatbot: ", response)