import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn 
from sklearn.ensemble import RandomForestClassifier

# STREAMLIT LAYOUT 
st.set_page_config(page_title="🎓 Student Dropout Prediction", layout="wide")

# LOAD MODEL AND EXPECTED COLUMNS 
@st.cache_resource
def load_model_and_columns():
    model = joblib.load("./scripts/dropout_retention_model_2.pkl")
    model_columns = joblib.load("./scripts/model_columns_2.pkl")
    return model, model_columns

def validate_category(col_name: str, mapped_value: str) -> str:
    """
    Jika dummy column f"{col_name}_{mapped_value}" tidak ada di model_columns,
    maka kembalikan "Other". 
    Otherwise return mapped_value.
    """
    print(f"mapped value : {mapped_value}")
    dummy_name = f"{col_name}_{mapped_value}"
    if dummy_name not in model_columns:
        return "Other"
    return mapped_value


model, model_columns = load_model_and_columns()


# REMAPPING ENCODED FEATURE CATEGORIES
marital_status_mapping = {
    "1": "Single", "2": "Married", "3": "Widower", "4": "Divorced",
    "5": "Facto Union", "6": "Legally Separated"
}

app_mode_mapping = {
    "1": "1st phase - general contingent",
    "2": "Ordinance No. 612/93",
    "5": "1st phase - special contingent (Azores Island)",
    "7": "Holders of other higher courses",
    "10": "Ordinance No. 854-B/99",
    "15": "International student (bachelor)",
    "16": "1st phase - special contingent (Madeira Island)",
    "17": "2nd phase - general contingent",
    "18": "3rd phase - general contingent",
    "26": "Ordinance No. 533-A/99, item b2 (Different Plan)",
    "27": "Ordinance No. 533-A/99, item b3 (Other Institution)",
    "39": "Over 23 years old",
    "42": "Transfer",
    "43": "Change of course",
    "44": "Technological specialization diploma holders",
    "51": "Change of institution/course",
    "53": "Short cycle diploma holders",
    "57": "Change of institution/course (International)"
}

course_mapping = {
    "33": "Biofuel Production Technologies",
    "171": "Animation and Multimedia Design",
    "8014": "Social Service (evening attendance)",
    "9003": "Agronomy",
    "9070": "Communication Design",
    "9085": "Veterinary Nursing",
    "9119": "Informatics Engineering",
    "9130": "Equinculture",
    "9147": "Management",
    "9238": "Social Service",
    "9254": "Tourism",
    "9500": "Nursing",
    "9556": "Oral Hygiene",
    "9670": "Advertising and Marketing Management",
    "9773": "Journalism and Communication",
    "9853": "Basic Education",
    "9991": "Management (evening attendance)"
}

prev_qual_mapping = {
    "1": "Secondary education",
    "2": "Higher education - bachelor's degree",
    "3": "Higher education - degree",
    "4": "Higher education - master's",
    "5": "Higher education - doctorate",
    "6": "Frequency of higher education",
    "9": "12th year of schooling - not completed",
    "10": "11th year of schooling - not completed",
    "12": "Other - 11th year of schooling",
    "14": "10th year of schooling",
    "15": "10th year of schooling - not completed",
    "19": "Basic education 3rd cycle (9th/10th/11th year) or equiv.",
    "38": "Basic education 2nd cycle (6th/7th/8th year) or equiv.",
    "39": "Technological specialization course",
    "40": "Higher education - degree (1st cycle)",
    "42": "Professional higher technical course",
    "43": "Higher education - master (2nd cycle)"
}

nationality_mapping = {
    "1": "Portuguese", "2": "German", "6": "Spanish", "11": "Italian",
    "13": "Dutch", "14": "English", "17": "Lithuanian", "21": "Angolan",
    "22": "Cape Verdean", "24": "Guinean", "25": "Mozambican",
    "26": "Santomean", "32": "Turkish", "41": "Brazilian",
    "62": "Romanian", "100": "Moldova (Republic of)", "101": "Mexican",
    "103": "Ukrainian", "105": "Russian", "108": "Cuban", "109": "Colombian"
}

qualification_mapping = {
    "1": "Secondary Education - 12th Year of Schooling or Eq.",
    "2": "Higher Education - Bachelor's Degree",
    "3": "Higher Education - Degree",
    "4": "Higher Education - Master's",
    "5": "Higher Education - Doctorate",
    "6": "Frequency of Higher Education",
    "9": "12th Year of Schooling - Not Completed",
    "10": "11th Year of Schooling - Not Completed",
    "11": "7th Year (Old)",
    "12": "Other - 11th Year of Schooling",
    "13": "2nd year complementary high school course",
    "14": "10th Year of Schooling",
    "18": "General commerce course",
    "19": "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.",
    "20": "Complementary High School Course",
    "22": "Technical-professional course",
    "25": "Complementary High School Course - not concluded",
    "26": "7th year of schooling",
    "27": "2nd cycle of the general high school course",
    "29": "9th Year of Schooling - Not Completed",
    "30": "8th year of schooling",
    "31": "General Course of Administration and Commerce",
    "33": "Supplementary Accounting and Administration",
    "34": "Unknown",
    "35": "Can't read or write",
    "36": "Can read without having a 4th year of schooling",
    "37": "Basic education 1st cycle (4th/5th year) or equiv.",
    "38": "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.",
    "39": "Technological specialization course",
    "40": "Higher education - degree (1st cycle)",
    "41": "Specialized higher studies course",
    "42": "Professional higher technical course",
    "43": "Higher Education - Master (2nd cycle)",
    "44": "Higher Education - Doctorate (3rd cycle)"
}

mothers_occupation_mapping = {
    "0": "Student",
    "1": "Legislative/Executive/Director/Manager",
    "2": "Intellectual & Scientific Activities",
    "3": "Intermediate Technicians & Professions",
    "4": "Administrative Staff",
    "5": "Personal Services/Security/Sellers",
    "6": "Farmers/Skilled Agriculture",
    "7": "Skilled Industry/Construction/Crafts",
    "8": "Machine Operators/Assembly Workers",
    "9": "Unskilled Workers",
    "10": "Armed Forces",
    "90": "Other Situation",
    "99": "(Blank)",
    "122": "Health Professionals",
    "123": "Teachers",
    "125": "ICT Specialists",
    "131": "Science/Engineering Techs",
    "132": "Intermediate Health Technicians",
    "134": "Legal/Social/Sports/Cultural Techs",
    "141": "Office Workers/Data Operators",
    "143": "Accounting/Financial Operators",
    "144": "Other Admin Support Staff",
    "151": "Personal Service Workers",
    "152": "Sellers",
    "153": "Personal Care Workers",
    "171": "Skilled Construction (not electricians)",
    "173": "Printing/Precision/Jewelry/Artisans",
    "175": "Food/Wood/Clothing Industries",
    "191": "Cleaning Workers",
    "192": "Unskilled Agriculture Workers",
    "193": "Unskilled Construction/Manufacturing",
    "194": "Meal Prep Assistants"
}

fathers_occupation_mapping = {
    "0": "Student",
    "1": "Legislative/Executive/Director/Manager",
    "2": "Intellectual & Scientific Activities",
    "3": "Intermediate Technicians & Professions",
    "4": "Administrative Staff",
    "5": "Personal Services/Security/Sellers",
    "6": "Farmers/Skilled Agriculture",
    "7": "Skilled Industry/Construction/Crafts",
    "8": "Machine Operators/Assembly Workers",
    "9": "Unskilled Workers",
    "10": "Armed Forces",
    "90": "Other Situation",
    "99": "(Blank)",
    "101": "Armed Forces Officers",
    "102": "Armed Forces Sergeants",
    "103": "Other Armed Forces Personnel",
    "112": "Admin/Commercial Service Directors",
    "114": "Hotel/Catering/Trade Directors",
    "121": "Physical Sciences/Engineering Specialists",
    "122": "Health Professionals",
    "123": "Teachers",
    "124": "Finance/Admin/Public Relations",
    "131": "Science/Engineering Technicians",
    "132": "Intermediate Health Technicians",
    "134": "Legal/Social/Sports/Cultural Techs",
    "135": "ICT Technicians",
    "141": "Office Workers/Data Operators",
    "143": "Accounting/Financial Operators",
    "144": "Other Admin Support Staff",
    "151": "Personal Service Workers",
    "152": "Sellers",
    "153": "Personal Care Workers",
    "154": "Security Services",
    "161": "Skilled Agricultural Workers",
    "163": "Subsistence Farmers/Fishers",
    "171": "Skilled Construction (not electricians)",
    "172": "Metalworking Workers",
    "174": "Electrical Workers",
    "175": "Food/Wood/Clothing Industries",
    "181": "Plant/Machine Operators",
    "182": "Assembly Workers",
    "183": "Vehicle/Mobile Equipment Operators",
    "192": "Unskilled Agriculture Workers",
    "193": "Unskilled Construction/Manufacturing",
    "194": "Meal Prep Assistants",
    "195": "Street Vendors/Service Providers"
}

daytime_evening_attendance_mapping = {
    "1": "Daytime",
    "0": "Evening"
}

binary_mapping = {
    "Displaced":   {"1": "Yes", "0": "No"},
    "Educational_special_needs": {"1": "Yes", "0": "No"},
    "Debtor":      {"1": "Yes", "0": "No"},
    "Gender":      {"1": "Male", "0": "Female"},
    "Scholarship_holder": {"1": "Yes", "0": "No"},
    "International": {"1": "Yes", "0": "No"},
    "Tuition_fees_up_to_date": {"1": "Fees up to date", "0": "Fees NOT up to date"}
}



st.title("🎓 Student Dropout Prediction App")

st.write("Fill in *all* fields below and click **Predict**.  \n"
         "→ If any field is missing, you will be prompted to complete it first.")


# INPUT WIDGETS 
with st.expander("Academic & Demographic Info", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        age_at_enrollment = st.number_input(
            label="Age at enrollment (years)", min_value=0, max_value=100, value=None
        )
        st.caption("e.g., 23, 18, etc.")
    with col2:
        marital_code = st.selectbox(
            "Marital status",
            options=list(marital_status_mapping.keys()),
            format_func=lambda x: marital_status_mapping[x],
            index=0
        )

    col3, col4 = st.columns(2)
    with col3:
        application_code = st.selectbox(
            "Application mode",
            options=list(app_mode_mapping.keys()),
            format_func=lambda x: app_mode_mapping[x],
            index=0
        )
    with col4:
        course_code = st.selectbox(
            "Course",
            options=list(course_mapping.keys()),
            format_func=lambda x: course_mapping[x],
            index=0
        )

    col5, col6 = st.columns(2)
    with col5:
        prev_qual_code = st.selectbox(
            "Previous qualification",
            options=list(prev_qual_mapping.keys()),
            format_func=lambda x: prev_qual_mapping[x],
            index=0
        )
    with col6:
        nationality_code = st.selectbox(
            "Nationality",
            options=list(nationality_mapping.keys()),
            format_func=lambda x: nationality_mapping[x],
            index=0
        )

    col7, col8 = st.columns(2)
    with col7:
        mothers_qual_code = st.selectbox(
            "Mother's qualification",
            options=list(qualification_mapping.keys()),
            format_func=lambda x: qualification_mapping[x],
            index=0
        )
    with col8:
        fathers_qual_code = st.selectbox(
            "Father's qualification",
            options=list(qualification_mapping.keys()),
            format_func=lambda x: qualification_mapping[x],
            index=0
        )

    col9, col10 = st.columns(2)
    with col9:
        mothers_occ_code = st.selectbox(
            "Mother's occupation",
            options=list(mothers_occupation_mapping.keys()),
            format_func=lambda x: mothers_occupation_mapping[x],
            index=0
        )
    with col10:
        fathers_occ_code = st.selectbox(
            "Father's occupation",
            options=list(fathers_occupation_mapping.keys()),
            format_func=lambda x: fathers_occupation_mapping[x],
            index=0
        )

    col11, col12 = st.columns(2)
    with col11:
        daytime_code = st.radio(
            "Daytime / evening attendance",
            options=list(daytime_evening_attendance_mapping.keys()),
            format_func=lambda x: daytime_evening_attendance_mapping[x],
            index=0
        )
    with col12:
        displaced_code = st.radio(
            "Displaced",
            options=list(binary_mapping["Displaced"].keys()),
            format_func=lambda x: binary_mapping["Displaced"][x],
            index=0
        )

    col13, col14 = st.columns(2)
    with col13:
        special_needs_code = st.radio(
            "Educational special needs",
            options=list(binary_mapping["Educational_special_needs"].keys()),
            format_func=lambda x: binary_mapping["Educational_special_needs"][x],
            index=0
        )
    with col14:
        debtor_code = st.radio(
            "Debtor",
            options=list(binary_mapping["Debtor"].keys()),
            format_func=lambda x: binary_mapping["Debtor"][x],
            index=0
        )

    col15, col16 = st.columns(2)
    with col15:
        gender_code = st.radio(
            "Gender",
            options=list(binary_mapping["Gender"].keys()),
            format_func=lambda x: binary_mapping["Gender"][x],
            index=0
        )
    with col16:
        scholarship_code = st.radio(
            "Scholarship holder",
            options=list(binary_mapping["Scholarship_holder"].keys()),
            format_func=lambda x: binary_mapping["Scholarship_holder"][x],
            index=0
        )

    col17, col18 = st.columns(2)
    with col17:
        international_code = st.radio(
            "International student",
            options=list(binary_mapping["International"].keys()),
            format_func=lambda x: binary_mapping["International"][x],
            index=0
        )
    with col18:
        tuition_fees_code = st.radio(
            "Tuition fees up to date",
            options=list(binary_mapping["Tuition_fees_up_to_date"].keys()),
            format_func=lambda x: binary_mapping["Tuition_fees_up_to_date"][x],
            index=0
        )
        
with st.expander("Academic History", expanded=False):
    app_order       = st.number_input("Application order", min_value=0, step=1, value=None)
    st.caption("(0 = first choice, 9 = last)")
    
    prev_qual_grade = st.number_input("Previous qualification grade", min_value=0.0, max_value=200.0, format="%.2f", value=None)
    st.caption("(0–200)")
    
    admission_grade = st.number_input("Admission grade", min_value=0.0, max_value=200.0, format="%.2f", value=None)
    st.caption("(0–200)")

with st.expander("Academic Performance", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        cu1_credit      = st.number_input("1st sem: Units credited", min_value=0, step=1, value=None)
        cu1_enrolled    = st.number_input("1st sem: Units enrolled", min_value=0, step=1, value=None)
        cu1_evaluations = st.number_input("1st sem: Units evaluated", min_value=0, step=1, value=None)
        cu1_approved    = st.number_input("1st sem: Units approved", min_value=0, step=1, value=None)
        cu1_grade       = st.number_input("1st sem: Avg grade", min_value=0.0, max_value=20.0, format="%.2f", value=None)
        cu1_without_ev  = st.number_input("1st sem: Units w/o evaluations", min_value=0, step=1, value=None)
        
    with col2:
        cu2_credit      = st.number_input("2nd sem: Units credited", min_value=0, step=1, value=None)
        cu2_enrolled    = st.number_input("2nd sem: Units enrolled", min_value=0, step=1, value=None)
        cu2_evaluations = st.number_input("2nd sem: Units evaluated", min_value=0, step=1, value=None)
        cu2_approved    = st.number_input("2nd sem: Units approved", min_value=0, step=1, value=None)
        cu2_grade       = st.number_input("2nd sem: Avg grade", min_value=0.0, max_value=20.0, format="%.2f", value=None)
        cu2_without_ev  = st.number_input("2nd sem: Units w/o evaluations", min_value=0, step=1, value=None)

    
with st.expander("Others", expanded=False):
    unemployment_rate = st.number_input("Unemployment rate (%)", min_value=0.0, max_value=100.0, format="%.2f", value=None)
    st.caption("(e.g., 5.7)")
    
    inflation_rate    = st.number_input("Inflation rate (%)", min_value=0.0, max_value=100.0, format="%.2f", value=None)
    st.caption("(e.g., 2.3)")
    
    gdp               = st.number_input("GDP (e.g. in € thousands)", min_value=0.0, format="%.2f", value=None)
    st.caption("(e.g., 12000.5)")


# PREDICTION BUTTON & LOGIC 
if st.button("🔍 Predict Dropout"):
    # check if all fields are filled
    missing = []
    # Check age separately
    if age_at_enrollment is None or age_at_enrollment == 0:
        missing.append("Age at enrollment")
    # Checking for missing variables
    for var_name, var_val in {
        "Marital status": marital_code,
        "Application mode": application_code,
        "Course": course_code,
        "Previous qualification": prev_qual_code,
        "Nationality": nationality_code,
        "Mother's qualification": mothers_qual_code,
        "Father's qualification": fathers_qual_code,
        "Mother's occupation": mothers_occ_code,
        "Father's occupation": fathers_occ_code,
        "Daytime/evening attendance": daytime_code,
        "Displaced": displaced_code,
        "Educational special needs": special_needs_code,
        "Debtor": debtor_code,
        "Gender": gender_code,
        "Scholarship holder": scholarship_code,
        "International": international_code,
        "Tuition fees up to date": tuition_fees_code,

        "1st sem units credited": cu1_credit,
        "1st sem units enrolled": cu1_enrolled,
        "1st sem units evaluated": cu1_evaluations,
        "1st sem units approved": cu1_approved,
        "1st sem avg grade": cu1_grade,
        "1st sem units w/o eval": cu1_without_ev,
        "2nd sem units credited": cu2_credit,
        "2nd sem units enrolled": cu2_enrolled,
        "2nd sem units evaluated": cu2_evaluations,
        "2nd sem units approved": cu2_approved,
        "2nd sem avg grade": cu2_grade,
        "2nd sem units w/o eval": cu2_without_ev,
        "Application order": app_order,
        "Prev qual grade": prev_qual_grade,
        "Admission grade": admission_grade,
        "Unemployment rate": unemployment_rate,
        "Inflation rate": inflation_rate,
        "GDP": gdp
    }.items():
        if var_val is None or (isinstance(var_val, (int, float)) and np.isnan(var_val)):
            missing.append(var_name)

    if missing:
        st.warning("⚠️ Please fill in all fields before predicting. Missing:\n\n- " + "\n- ".join(missing))
    else:
        # RAW ROW DICTIONARY
        row = {
            "Age_at_enrollment": age_at_enrollment,
            "Marital status": marital_status_mapping[marital_code],
            "Application mode": app_mode_mapping[application_code],
            "Course": course_mapping[course_code],
            "Previous qualification": prev_qual_mapping[prev_qual_code],
            "Nacionality": validate_category("Nacionality", nationality_mapping[nationality_code]),
            "Mother's qualification": qualification_mapping[mothers_qual_code],
            "Father's qualification": qualification_mapping[fathers_qual_code],
            "Mother's occupation": mothers_occupation_mapping[mothers_occ_code],
            "Father's occupation": fathers_occupation_mapping[fathers_occ_code],
            "Daytime/evening attendance": daytime_evening_attendance_mapping[daytime_code],
            "Displaced": binary_mapping["Displaced"][displaced_code],
            "Educational special needs": binary_mapping["Educational_special_needs"][special_needs_code],
            "Debtor": binary_mapping["Debtor"][debtor_code],
            "Gender": binary_mapping["Gender"][gender_code],
            "Scholarship holder": binary_mapping["Scholarship_holder"][scholarship_code],
            "International": binary_mapping["International"][international_code],
            "Tuition_fees_up_to_date": binary_mapping["Tuition_fees_up_to_date"][tuition_fees_code],

            # Numeric fields:
            "Curricular_units_1st_sem_credited": cu1_credit,
            "Curricular_units_1st_sem_enrolled": cu1_enrolled,
            "Curricular_units_1st_sem_evaluations": cu1_evaluations,
            "Curricular_units_1st_sem_approved": cu1_approved,
            "Curricular_units_1st_sem_grade": cu1_grade,
            "Curricular_units_1st_sem_without_evaluations": cu1_without_ev,

            "Curricular_units_2nd_sem_credited": cu2_credit,
            "Curricular_units_2nd_sem_enrolled": cu2_enrolled,
            "Curricular_units_2nd_sem_evaluations": cu2_evaluations,
            "Curricular_units_2nd_sem_approved": cu2_approved,
            "Curricular_units_2nd_sem_grade": cu2_grade,
            "Curricular_units_2nd_sem_without_evaluations": cu2_without_ev,

            "Application_order": app_order,
            "Previous_qualification_grade": prev_qual_grade,
            "Admission_grade": admission_grade,

            "Unemployment_rate": unemployment_rate,
            "Inflation_rate": inflation_rate,
            "GDP": gdp
        }

        df_raw = pd.DataFrame([row])

        # create AgeGroup
        bins = [0, 20, 24, 30, 100]
        labels = ['<=20', '21-24', '25-30', '31+']
        df_raw['AgeGroup'] = pd.cut(
            df_raw['Age_at_enrollment'], bins=bins, labels=labels, right=False
        )

        # drop unused column
        df_raw = df_raw.drop(columns=['Age_at_enrollment'])

        # ONE‐HOT ENCODE ALL CATEGORICALS 
        cat_cols = df_raw.select_dtypes(include=['object', 'category']).columns
        df_encoded = pd.get_dummies(df_raw, columns=cat_cols)

        # reindex to match columns used during training
        df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)
        
        # df_encoded.info()
        
        # df_encoded.to_excel('one_row_df.xlsx',index=False)

        # PREDICT & DISPLAY
        pred = model.predict(df_encoded)[0]
        # target col 'Is_Dropout': 1=Dropout, 0=Not_Dropout
        result_str = "✅ **Not Dropped Out**" if pred == 0 else "❌ **Will Drop Out**"
        st.success(f"Prediction: {result_str}")
