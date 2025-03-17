from string_commonalities import find_nonoverlapping_common_substrings
import streamlit as st, json, os, random
from collections import defaultdict
from datetime import datetime

# annotator_assignment = {
#     "celeste": [f"creative_writing_triplet_{i}" for i in [1, 2, 3, 5, 6, 7, 8, 9, 10, 11]],
#     "harry": [f"creative_writing_triplet_{i}" for i in [12, 14, 15, 16, 17, 18, 19, 20, 21, 22]],
#     "philippe": [f"creative_writing_triplet_{i}" for i in range(5,15)],
#     "tuhin": [f"creative_writing_triplet_{i}" for i in range(5,10)],    
# }
with open("data/annotations/annotator_assignment.json", "r") as f:
    annotator_assignment = json.load(f)

# Set page configuration
st.set_page_config(
    page_title="Writing Sample Annotation",
    layout="wide",
    initial_sidebar_state="expanded"
)

annotation_sample_file = "data/preference_annotation_triplets_fixed2.jsonl"

# Load the dataset
# @st.cache_data
def load_dataset():
    with open(annotation_sample_file, "r") as f:
        return [json.loads(line) for line in f]

# Get counts of annotations per sample
def get_annotation_counts():
    counts = defaultdict(int)
    if not os.path.exists("data/annotations"):
        os.makedirs("data/annotations")
    
    for filename in os.listdir("data/annotations"):
        if filename.endswith(".jsonl"):
            with open(f"data/annotations/{filename}", "r") as f:
                for line in f:
                    try:
                        annotation = json.loads(line)
                        counts[annotation["sample_id"]] += 1
                    except:
                        pass
    return counts

# Get annotator's completed annotations
def get_annotator_completed(annotator_name):
    completed = set()
    annotation_file = f"data/annotations/{annotator_name}.jsonl"
    
    if os.path.exists(annotation_file):
        with open(annotation_file, "r") as f:
            for line in f:
                try:
                    annotation = json.loads(line)
                    completed.add(annotation["sample_id"])
                except:
                    pass
    return completed

# Save annotation
def save_annotation(annotator_name, sample_id, rankings):
    annotation = {
        "annotator": annotator_name,
        "sample_id": sample_id,
        "rankings": rankings,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(f"data/annotations/{annotator_name}.jsonl", "a") as f:
        f.write(json.dumps(annotation) + "\n")

# Function to highlight common parts across all three texts
def highlight_common_parts(texts):
    # Find common substrings across all three texts
    common_parts = find_nonoverlapping_common_substrings(texts[0], texts[1], texts[2])
    
    # Create highlighted versions of each text
    highlighted_texts = []
    
    for text in texts:
        result = text
        # Replace each common part with an underlined version
        for common_part in common_parts:
            result = result.replace(common_part, f'<span style="text-decoration: underline;">{common_part}</span>')
        
        highlighted_texts.append(result)
    
    return highlighted_texts

# Main app
def main():
    # Initialize session state for annotator name
    if "annotator_name" not in st.session_state:
        st.session_state.annotator_name = ""
    
    # Login screen
    if not st.session_state.annotator_name:
        st.title("Writing Sample Quality Judgment")
        st.write("Please enter your name to access the writing samples.")
        
        with st.form("login_form"):
            annotator_name = st.text_input("Name (alphanumeric characters only)")
            submit_button = st.form_submit_button("Start")
            
            if submit_button:
                if annotator_name and annotator_name.isalnum():
                    st.session_state.annotator_name = annotator_name.lower()  # Convert to lowercase
                    st.rerun()
                else:
                    st.error("Please enter a valid name using only letters and numbers (no spaces or special characters).")
    
    # Annotation interface
    else:
        dataset = load_dataset()
        annotation_counts = get_annotation_counts()
        completed_samples = get_annotator_completed(st.session_state.annotator_name)
        
        # Top bar with user info and logout
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.write(f"**Annotator:** {st.session_state.annotator_name}")
        with col2:
            st.write(f"**Completed:** {len(completed_samples)} judgments")
        with col3:
            if st.button("Logout"):
                st.session_state.annotator_name = ""
                st.rerun()
        
        st.markdown("---")
        
        # Main panel
        st.title("Writing Sample Judgment")
        
        # Get samples assigned to this annotator (use lowercase for lookup)
        assigned_sample_ids = annotator_assignment.get(st.session_state.annotator_name.lower(), [])
        
        if not assigned_sample_ids:
            st.warning(f"No samples have been assigned to {st.session_state.annotator_name}. Please contact the administrator.")
            return
        
        # Find assigned samples not yet annotated by this annotator
        available_samples = [s for s in dataset if s["id"] in assigned_sample_ids and s["id"] not in completed_samples]
        
        if not available_samples:
            st.success("You have completed all assigned samples. Thank you!")
            return
        
        # Sort by number of annotations (prioritize those with fewer annotations)
        available_samples.sort(key=lambda s: annotation_counts[s["id"]])
        current_sample = available_samples[0]
        
        # Display the instruction
        st.header("Writing Instruction:")
        st.write(current_sample["instruction"]['plot'])
        st.markdown("---")
        
        # Prepare the three candidates in random order with newlines replaced
        raw_texts = [
            current_sample["first_draft"]["candidate"].replace("\n", " "),
            current_sample["random_cot"]["candidate"].replace("\n", " "),
            current_sample["best_cot"]["candidate"].replace("\n", " ")
        ]
        
        # Highlight common parts
        highlighted_texts = highlight_common_parts(raw_texts)
        
        candidates = [
            {"id": "first_draft", "text": highlighted_texts[0]},
            {"id": "random_cot", "text": highlighted_texts[1]},
            {"id": "best_cot", "text": highlighted_texts[2]}
        ]
        
        # Store the shuffled candidates in session state to maintain order between reruns
        if "shuffled_candidates" not in st.session_state or st.session_state.get("current_sample_id") != current_sample["id"]:
            random.shuffle(candidates)
            st.session_state.shuffled_candidates = candidates
            st.session_state.current_sample_id = current_sample["id"]
        else:
            candidates = st.session_state.shuffled_candidates
        
        # Create form for ranking with a unique key based on the sample ID
        with st.form(key=f"ranking_form_{current_sample['id']}"):
            st.subheader("Please rank these writing samples from most to least preferred:")
            
            cols = st.columns(3)
            
            # Initialize rankings dictionary in session state if not present
            if "rankings" not in st.session_state:
                st.session_state.rankings = {}
            
            for i, (col, candidate) in enumerate(zip(cols, candidates)):
                with col:
                    st.markdown(f"**Sample {i+1}**")
                    st.markdown(f"""
                    <div style="border:1px solid #ddd; padding:10px; height:400px; overflow-y:auto; background-color:#f9f9f9;">
                    {candidate["text"]}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Use a key that includes both the sample ID and candidate ID
                    rank = st.radio(
                        f"Rank Sample {i+1}",
                        ["Most Preferred Writing Option", "Second Favorite Writing Option", "Least Preferred Writing Option"],
                        key=f"rank_{current_sample['id']}_{candidate['id']}"
                    )
                    
                    # Store the ranking in session state
                    st.session_state.rankings[candidate["id"]] = rank
            
            submit = st.form_submit_button("Submit Rankings")
            
            if submit:
                # if they submit but there's not ranking, or annotator name, then throw an error
                if len(st.session_state.rankings) == 0 or st.session_state.annotator_name == "":
                    st.error("There was an error, please log out / log in. Sorry!")
                # Check if all rankings are unique
                elif len(set(st.session_state.rankings.values())) < 3:
                    st.error("Please assign a unique rank to each sample.")
                else:
                    print(">>", st.session_state.annotator_name, current_sample["id"], st.session_state.rankings)
                    # save to backup_annotation.txt
                    with open("data/backup_annotation.txt", "a") as f:
                        f.write(f"{st.session_state.annotator_name} {current_sample['id']} {st.session_state.rankings}\n")
                    save_annotation(st.session_state.annotator_name, current_sample["id"], st.session_state.rankings)
                    st.success("Annotation saved successfully!")
                    # Clear the rankings for the next sample
                    st.session_state.rankings = {}
                    # Remove the shuffled candidates to get a new shuffle for the next sample
                    del st.session_state.shuffled_candidates
                    st.rerun()

if __name__ == "__main__":
    main()