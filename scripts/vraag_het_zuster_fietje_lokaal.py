import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def load_model_and_tokenizer(model_id: str, tokenizer_id: str):
    """
    Load the pre-trained fine-tuned model and tokenizer.

    Args:
        model_id (str): The identifier for the pre-trained model.
        tokenizer_id (str): The identifier for the tokenizer.

    Returns:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
    """
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    return model, tokenizer


def tokenize_prompt(model, prompt):
    # Tokenize and prepare input
    return tokenizer(prompt, return_tensors="pt").input_ids.to(model.device), tokenizer(
        prompt, return_tensors="pt", padding=True
    ).attention_mask.to(model.device)


def generate_output(model, input_ids, attention_mask):
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            do_sample=True,
            top_p=1,
            top_k=50,
            temperature=0.1,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def answer(model, prompt):
    input_ids, attention_mask = tokenize_prompt(model, prompt)
    generated_text = generate_output(
        model=model, input_ids=input_ids, attention_mask=attention_mask
    )
    return generated_text[len(prompt) :].strip()


def answer_row(row):
    def create_prompt(row: dict, add_response: bool = True) -> str:
        """
        Generates a prompt based on the input data in 'row'.

        Args:
            row (dict): A dictionary containing 'context', 'instruction', and optionally 'response'.
            add_response (bool): If True, the prompt will include the 'response'.
                                 If False, only 'context' and 'instruction' will be included.

        Returns:
            str: The generated prompt in text format.
        """
        # Base prompt (without response)
        prompt = f"""|CONTEXT|
{row.get('context', '')}

|INSTRUCTION|
{row.get('instruction', '')}

|RESPONSE|
"""

        # Append response if 'add_response' is True and 'response' exists
        if add_response and "response" in row:
            prompt += f"\n{row['response']}\n"

        return prompt

    # Prepare the prompt with notes from sample
    prompt = create_prompt(row=row, add_response=False)
    print(prompt)

    # Display the generated response and actual response
    ref_response = row["response"]  # Reference response from dataset
    print("\nREFERENCE RESPONSE:")
    print(ref_response)
    print(f"\n{100*'-'}")
    print("ZUSTER FIETJE:")
    print(answer(model, prompt))
    print(f"\n{100*'-'}")


# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer(
    model_id="/Users/eva/Library/CloudStorage/GoogleDrive-e.k.rombouts@gmail.com/My Drive/results_full/checkpoint-4600",
    tokenizer_id="BramVanroy/fietje-2-instruct",
)
# Load dataset
dataset = load_dataset(path="ekrombouts/Gardenia_instruct_dataset", split="validation")

answer_row(dataset[3])

prompt = """|CONTEXT|

U was inc van urine. U was niet vriendelijk tijdens het verschonen.
Mw was vanmorgen incontinent van dunne def, bed was ook nat. Mw is volledig verzorgd, bed is verschoond,
Mw. haar kledingkast is opgeruimd.
Mw. zei:"oooh kind, ik heb zo'n pijn. Mijn benen. Dat gaat nooit meer weg." Mw. zat in haar rolstoel en haar gezicht trok weg van de pijn en kreeg traanogen. Mw. werkte goed mee tijdens adl. en was vriendelijk aanwezig. Pijn. Mw. kreeg haar medicatie in de ochtend, waaronder pijnstillers. 1 uur later adl. gegeven.
Ik lig hier maar voor Piet Snot. Mw. was klaarwakker tijdens eerste controle. Ze wilde iets, maar wist niet wat. Mw. een slokje water gegeven en uitgelegd hoe ze kon bellen als ze iets wilde. Mw. pakte mijn hand en bedankte me.
Mevr. in de ochtend ondersteund met wassen en aankleden. Mevr was rustig aanwezig.
Mw is volledig geholpen met ochtendzorg, mw haar haren zijn gewassen. Mw haar nagels zijn kort geknipt.
Mevr heeft het ontbijt op bed genuttigd. Daarna mocht ik na de tweede poging Mevr ondersteunen met wassen en aankleden.
Vanmorgen met mw naar buiten geweest om een sigaret te roken. Mw was niet erg spraakzaam en mw kwam op mij over alsof ze geen behoefte had aan een gesprek. Mw kreeg het koud door de wind en wilde snel weer naar binnen.

|INSTRUCTION|
Geef de drie belangrijkste lichamelijke klachten van mw.

|RESPONSE|
"""

print(answer(model, prompt))

prompt = """|CONTEXT|
Dhr kreeg zijn avondploeg, was zeer behulpzaam tijdens de zorg
U was vandaag vriendelijk in de omgang. Toen ik u voor de lunch kwam halen, kreeg ik zelfs een glimlach van u
S: Bedankt voor je hulp.O:Dhr is geholpen met scheren, dhr is gedoucht en bed is verschoond. Dhr was over het algemeen behulpzaam.A:Ochtend zorg
Dhr. begeleid naar het toilet met resultaat van def. Zie metingen
Ochtendmedicatie zaterdag is binnengekomen. Ligt in de medicatiekar
S/ Tc zorgDhr. heeft gisteren dubbele ochtendmedicatie gehad. clopidrogel 75 mg en foliumzuur 0,5 mg keer twee dus. P/- Medicatie bestellen zodat je genoeg hebt voor het weekend bij de apotheek- vim melding maken- Arts Jan Jansen gevraagd voor advies, geeft aan dat het niet uit maakt
Apotheek is gebeld. Medicatie voor zaterdag wordt vanavond geleverd.
Vim ingevuld. Meneer kreeg vanmorgen zijn medicatie, dit werd ook afgetekend door de zorg. Bij controle bleek dat de zaterdag van de rol was. De zorg dacht dat meneer vandaag dan dubbele medicatie had gekregen. Er is navraag gedaan bij de collega die op het achterom stond of deze medicatie had gegeven aan meneer. Dit was niet het geval. Meneer heeft waarschijnlijk op donderdag dubbele medicatie gekregen. 8112 is gebeld en besproken, vm 1e contactpersoon ingesproken
Meneer is vanmorgen begeleid met de adl.
S:O:U was wakker tijdens de controle van u inco. A:P:U bent verschoond en gedraaid op u Li-zij.

|INSTRUCTION|
Geef de belangrijkste lichamelijke klachten van dhr.

|RESPONSE|
"""

print(answer(model, prompt))