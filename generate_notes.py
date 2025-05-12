import argparse
import json
import os
import sys
import openai
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
print("Loading environment variables...", file=sys.stderr)
load_dotenv()

# Get the OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY not found. Make sure it's set in your .env file.", file=sys.stderr)
    exit(1) # Exit the script if the key is missing
else:
    masked_key = OPENAI_API_KEY[:5] + "..." + OPENAI_API_KEY[-4:]
    print(f"OpenAI API Key loaded (masked): {masked_key}", file=sys.stderr)


def generate_discharge_note(consultation_data_str):
    """
    Uses OpenAI LLM to generate a discharge note from consultation data string.
    """
    print("Attempting to call OpenAI API...", file=sys.stderr)
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)

        system_prompt = """
        You are a helpful veterinary assistant.
        You will be provided with a summary of a veterinary consultation. 
        Your task is to generate a clear and concise discharge note TEXT for the pet owner.
        Consider the following criteria:
        - The note should be easy to understand for a non-medical person.
        - Address the pet owner directly (e.g., "Regarding your pet, [PetName], ...").
        - Summarize key findings, treatments performed (if any), and any medications 
        to administer at home (with dosage and frequency if provided).
        - If no specific medications are listed for home care, state that.
        - Include important next steps, follow-up advice, or things to monitor.
        - Make the tone caring and supportive.
        - Avoid using medical jargon or complex terminology.
        - Ensure the note is well-structured and easy to read.
        - If there are any specific instructions or observations from the vet,
        include them in the note.
        Provide the note in a friendly and professional manner.
        Output ONLY the discharge note text, without any additional comments or explanations.
        """

        # The consultation_data_str will be the pre-processed text summary of the JSON data
        user_prompt = f"""
        Please generate the text for a discharge note based on the following consultation information:
        ---
        {consultation_data_str}
        ---
        Remember to focus on what the pet owner needs to know and do.
        """

        print("User prompt sent to OpenAI", file=sys.stderr)
        print(user_prompt, file=sys.stderr)
        print("End user prompt", file=sys.stderr)

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=700,
        )
        print("OpenAI API call completed.", file=sys.stderr)

        raw_content = completion.choices[0].message.content.strip()
        print(f"Raw content from OpenAI: \n{raw_content}\n End raw API response", file=sys.stderr)

        return raw_content.strip()

    except openai.APIConnectionError as e:
        print(f"Error: Could not connect to OpenAI API. {e}", file=sys.stderr)
        return None
    except openai.RateLimitError as e:
        print(f"Error: Rate limit exceeded. {e}", file=sys.stderr)
        return None
    except openai.AuthenticationError as e:
        print(f"Error: Authentication failed. {e}", file=sys.stderr)
        return None
    except openai.APIError as e:
        print(f"Error: OpenAI API error. {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        return None

# Main function to handle the script execution
def main():
    # Argument Parsing
    parser = argparse.ArgumentParser(description="Generate discharge notes from veterinary consultation JSON data.")
    parser.add_argument("json_file_path", type=str, help="Path to the consultation JSON file.")
    args = parser.parse_args()

    input_file_path = args.json_file_path
    print(f"Attempting to load JSON from: {input_file_path}", file=sys.stderr)

    # Load and Parse Input JSON 
    try:
        with open(input_file_path, 'r') as f:
            consultation_data = json.load(f)
        print(f"Loaded JSON data successfully from {input_file_path}", file=sys.stderr)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file_path}", file=sys.stderr)
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_file_path}. Please check the file format.", file=sys.stderr)
        exit(1)

    # Debug: Print loaded JSON data 
    print("--- Loaded Consultation Data ---", file=sys.stderr)
    print(json.dumps(consultation_data, indent=2), file=sys.stderr) # Print entire input data for debugging
    print("--- End Loaded Consultation Data ---", file=sys.stderr)
    # End Debug 

    # Extract Relevant Data from JSON and Prepare for LLM 
    # Create a string summary of the consultation data to feed to the LLM
    patient_info = consultation_data.get("patient", {})
    consult_info = consultation_data.get("consultation", {})
    treatment_items = consult_info.get("treatment_items", {})

    patient_name = patient_info.get("name", "Your pet")
    patient_species = patient_info.get("species", "the species")
    consult_date = consult_info.get("date", "the recent visit")
    consult_reason = consult_info.get("reason", "the consultation")

    # Clinical Notes
    clinical_notes_list = consult_info.get("clinical_notes", [])
    clinical_notes_summary = "No specific clinical notes recorded for this visit."
    if clinical_notes_list:
        notes_texts = [note.get("note", "") for note in clinical_notes_list if note.get("note")]
        if notes_texts:
            clinical_notes_summary = "Clinical observations: " + "\n".join(notes_texts)

    # Treatment items extraction 
    treatment_details_parts = [] 

    # Procedures
    procedures_list = treatment_items.get("procedures", [])
    if procedures_list:
        proc_names = [proc.get("name", "Unnamed Procedure") for proc in procedures_list if proc.get("name")]
        if proc_names:
            treatment_details_parts.append("Procedures performed: " + ", ".join(proc_names) + ".")

    # Medicines 
    medicines_list = treatment_items.get("medicines", [])
    if medicines_list:
        med_summaries = []
        for med in medicines_list:
            med_name = med.get("name", "Unnamed Medicine")
            # Real medicine data would have dosage, frequency, duration.
            # Example formatting if such fields exist:
            # dosage = med.get("dosage", "")
            # frequency = med.get("frequency", "")
            # duration = med.get("duration", "")
            # detail_string = f"{med_name}"
            # if dosage: detail_string += f" (Dosage: {dosage}"
            # if frequency: detail_string += f", Frequency: {frequency}"
            # if duration: detail_string += f", Duration: {duration}"
            # if dosage: detail_string += ")" # close parenthesis if dosage was present
            # med_summaries.append(detail_string)

            # For now, with current JSON structure, just the name:
            med_summaries.append(med_name)
        if med_summaries:
            treatment_details_parts.append("Medicines administered during the visit: " + ", ".join(med_summaries) + ".")

    # Prescriptions 
    prescriptions_list = treatment_items.get("prescriptions", [])
    if prescriptions_list:
        home_med_details = []
        for prescr in prescriptions_list:
            prescr_name = prescr.get("name", "Unnamed Prescription")
            # Real prescription data would have dosage, frequency, duration.
            # Example of formatting if such fields exist:
            # dosage = prescr.get("dosage", "")
            # frequency = prescr.get("frequency", "")
            # duration = prescr.get("duration", "")
            # detail_string = f"{prescr_name}"
            # if dosage: detail_string += f" (Dosage: {dosage}"
            # if frequency: detail_string += f", Frequency: {frequency}"
            # if duration: detail_string += f", Duration: {duration}"
            # if dosage: detail_string += ")" # close parenthesis if dosage was present
            # home_med_details.append(detail_string)

            # For now, with current JSON structure, just the name:
            home_med_details.append(prescr_name)

        if home_med_details:
            treatment_details_parts.append("Medications prescribed for home care: " + ", ".join(home_med_details) + ". Please follow the specific instructions provided for administration.")
    else:
        # Explicitly state if no new prescriptions
        treatment_details_parts.append("Medications prescribed for home care: No new medications were prescribed for you to take home during this visit. If your pet is on existing medication, please continue as previously directed.")


    # Foods 
    foods_list = treatment_items.get("foods", [])
    if foods_list:
        food_names = [food.get("name", "Unnamed Food Item") for food in foods_list if food.get("name")]
        if food_names:
            treatment_details_parts.append("Specific foods recommended/dispensed: " + ", ".join(food_names) + ".")

    # Supplies 
    supplies_list = treatment_items.get("supplies", [])
    if supplies_list:
        supply_names = [supply.get("name", "Unnamed Supply") for supply in supplies_list if supply.get("name")]
        if supply_names:
            treatment_details_parts.append("Supplies provided: " + ", ".join(supply_names) + ".")

    # Join all treatment details into one block
    all_treatment_summary = "\n\n".join(treatment_details_parts) if treatment_details_parts else "No specific treatments, medications, or supplies were recorded for this visit."

    # Diagnostics
    diagnostics_list = consult_info.get("diagnostics", [])
    diagnostics_summary = "Diagnostics performed: No diagnostics were recorded for this visit." # Default
    if diagnostics_list:
        diag_names = [diag.get("name", "Unnamed Diagnostic") for diag in diagnostics_list if diag.get("name")]
        if diag_names:
            diagnostics_summary = "Diagnostics performed: " + ", ".join(diag_names) + "."

    # Construct the string for the LLM
    consultation_summary_for_llm = f"""
    Patient Name: {patient_name}
    Species: {patient_species}
    Consultation Date: {consult_date}
    Reason for Visit: {consult_reason}

    {clinical_notes_summary}

    --- Diagnostics Summary ---
    {diagnostics_summary}
    --- End Diagnostics Summary --

    --- Treatment Summary ---
    {all_treatment_summary}
    --- End Treatment Summary ---

    Other Instructions/Observations from Vet: (Relying on LLM to infer general advice based on the provided data.)
    """

    print(" Summary string prepared for LLM: ", file=sys.stderr)
    print(consultation_summary_for_llm, file=sys.stderr)
    print(" End summary string for LLM", file=sys.stderr)

    # Call the LLM to Generate the Discharge Note
    generated_note_text = generate_discharge_note(consultation_summary_for_llm)

    if generated_note_text is None:
        print("Failed to generate discharge note.", file=sys.stderr)
        exit(1)
    elif not generated_note_text:
        print("Warning: Generated note is empty.", file=sys.stderr)
        exit(1)
    print("Discharge note generated successfully.", file=sys.stderr)
    print("Generated Discharge Note: ", file=sys.stderr)
    print(generated_note_text, file=sys.stderr)
    print("End generated discharge note", file=sys.stderr)

    # Debug: Print generated note text
    print(f"Sending data to LLM for {patient_name}...", file=sys.stderr)

    # Format and Print Output JSON
    output_json = {
        "discharge_note": generated_note_text
    }

    # Print the JSON to standard output 
    print("Preparing final JSON output to stdout...", file=sys.stderr)
    print(json.dumps(output_json, indent=2))
    print("Script finished.", file=sys.stderr)


if __name__ == "__main__":
    main()