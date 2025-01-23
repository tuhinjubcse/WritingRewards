import streamlit as st, json, pandas as pd, random
from utils_diff import make_colored_text

random.seed(42)

# Load the data
@st.cache_data
def load_data():
    with open("data/lamp_editing_benchmark.json", "r") as f:
        return json.load(f)

def main():
    st.title("LAMP Editing Benchmark Visualization")
    
    # Load data
    eval_samples = load_data()
    
    # Sample selector
    sample_idx = st.selectbox(
        "Select sample", 
        range(len(eval_samples)),
        format_func=lambda x: f"Sample {x+1}"
    )
    
    sample = eval_samples[sample_idx]
    
    # Display original text
    st.subheader("Original Text")

    ai_draft_candidate = [c for c in sample["candidates"] if c["system"] == "ai_draft"][0]
    st.write(ai_draft_candidate["text"])
    
    # Display each candidate's diff
    st.subheader("Candidate Edits")
    
    random.shuffle(sample["candidates"])

    for i, candidate in enumerate(sample["candidates"]):
        with st.expander(f"Candidate {i+1}", expanded=True):
            if candidate["system"] == "ai_draft":
                continue
            # Create colored diff
            diff_html = make_colored_text(
                ai_draft_candidate["text"].strip(), 
                candidate["text"].strip(), 
                style="html"
            )
            
            # Add CSS for the diff colors
            # the deletion should also have a strike-through effect
            st.markdown("""
                <style>
                .green { background-color: #90EE90; }
                .red { background-color: #FFB6C1; text-decoration: line-through; }
                .blue { background-color: #ADD8E6; }
                </style>
                """, unsafe_allow_html=True)
            
            # Display the diff
            st.markdown(diff_html, unsafe_allow_html=True)
            
            # Reveal system and score button
            if st.button(f"Reveal Details for Candidate {i+1}"):
                st.write(f"System: {candidate['system']}")
                st.write(f"Reward Score: {candidate.get('score', -1.0):.3f}")
    
    # Display pairwise comparisons
    st.subheader("Pairwise Comparisons")
    
    # Create a matrix of pairwise preferences
    systems = [c["system"] for c in sample["candidates"]]
    pairwise_matrix = pd.DataFrame(index=systems, columns=systems)
    
    for key, value in sample["pairwise_prefs"].items():
        sys1, sys2 = key.split("__")
        if value == 1:
            result = f"{sys1} preferred"
        elif value == 2:
            result = f"{sys2} preferred"
        else:
            result = "Tie"
        pairwise_matrix.loc[sys1, sys2] = result
        pairwise_matrix.loc[sys2, sys1] = result
    
    st.dataframe(pairwise_matrix)

if __name__ == "__main__":
    main() 