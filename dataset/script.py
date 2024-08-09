import random
import pandas as pd
from faker import Faker

fake = Faker()

def generate_probabilities():
    # Determine which category will have the high probability
    high_prob_category = random.choice(["normal", "mci", "ad"])

    if high_prob_category == "normal":
        normal = round(random.uniform(97, 99.9), 2)
        remaining = 100 - normal
        mci = round(random.uniform(0, remaining), 2)
        ad = round(100 - normal - mci, 2)
    elif high_prob_category == "mci":
        mci = round(random.uniform(97, 99.9), 2)
        remaining = 100 - mci
        normal = round(random.uniform(0, remaining), 2)
        ad = round(100 - mci - normal, 2)
    else:  # high_prob_category == "ad"
        ad = round(random.uniform(97, 99.9), 2)
        remaining = 100 - ad
        normal = round(random.uniform(0, remaining), 2)
        mci = round(100 - ad - normal, 2)

    return normal, mci, ad

def random_sentences(sentences, num=1):
    return ' '.join(random.sample(sentences, num))

def generate_medical_report(normal, mci, ad):
    risk_level = "low" if normal > 90 else "moderate" if normal > 70 else "high"

    intro_styles = [
        lambda s: f"Medical Report Summary:\n\n{s}",
        lambda s: f"Cognitive Health Evaluation:\n\n{s}",
        lambda s: f"Neurocognitive Assessment Results:\n\n{s}",
        lambda s: f"Brain Health Analysis:\n\n{s}",
        lambda s: f"Cognitive Function Report:\n\n{s}"
    ]

    intro_sentences = [
        "After a comprehensive analysis of the patient genetic testing results, MRI scan, and health records, we've developed the following cognitive profile.",
        "A thorough review of available data, including genetic markers, neuroimaging, and medical history, provides the following insights into the patient cognitive status.",
        "Based on an in-depth examination of the patient genetic predisposition, brain imaging, and overall health information, we can draw these conclusions about their cognitive health.",
        "Utilizing advanced analytical techniques on the patient genetic, neuroimaging, and health data, we've constructed the following cognitive assessment.",
        "Our multidisciplinary team has carefully evaluated the patient genetic markers, brain scans, and medical records to produce this cognitive health summary.",
        "Integrating cutting-edge genetic analysis, advanced neuroimaging techniques, and comprehensive health records, we present the following cognitive profile.",
        "After a meticulous review of the patient biomarkers, brain structure, and medical history, we've synthesized the following cognitive health assessment."
        "Upon meticulous examination of the patient neurogenetic markers, cerebral imaging, and comprehensive health dossier, we present the following cognitive health synopsis.",
        "Leveraging state-of-the-art analytical methodologies, we've scrutinized the patient genetic predispositions, neuroanatomical structures, and medical chronicles to formulate this cognitive assessment.",
        "Through the lens of advanced neuroscience and personalized medicine, we've synthesized insights from genomic data, brain architecture, and clinical history to produce this cognitive health profile.",
        "Employing a multifaceted approach integrating genetic risk factors, neuroimaging biomarkers, and longitudinal health data, we've constructed the following neurocognitive evaluation.",
        "By harmonizing cutting-edge genetic analysis, high-resolution brain imaging, and extensive medical records, we've crafted a nuanced portrait of the patient cognitive landscape.",
        "Drawing upon the latest advancements in neurocognitive science, we've amalgamated genetic indicators, structural brain data, and comprehensive health metrics to generate this cognitive health report.",
        "Through the synergistic interpretation of genomic vulnerabilities, neuroanatomical peculiarities, and holistic health parameters, we present this detailed cognitive function assessment."
    ]

    probability_styles = [
        lambda s: f"Probability Distribution:\n{s}",
        lambda s: f"Cognitive Risk Stratification:\n{s}",
        lambda s: f"Neurocognitive Status Probabilities:\n{s}",
        lambda s: f"Brain Health Risk Assessment:\n{s}",
        lambda s: f"Cognitive Function Likelihood Analysis:\n{s}"
        lambda s: f"Risk Probability Matrix:\n{s}",
        lambda s: f"Outcome Probabilities:\n{s}"
    ]

    probability_sentences = [
        f"Our analysis indicates a {normal}% probability of normal cognitive function, a {mci}% likelihood of Mild Cognitive Impairment, and a {ad}% risk of Alzheimer Disease.",
        f"The cognitive profile reveals a {normal}% chance of typical brain function, a {mci}% possibility of Mild Cognitive Impairment, and a {ad}% potential for Alzheimer Disease.",
        f"Quantitative assessment suggests {normal}% likelihood of cognitive normalcy, {mci}% probability of Mild Cognitive Impairment, and {ad}% risk factor for Alzheimer Disease.",
        f"Statistical analysis of the data points to a {normal}% probability of normal cognition, a {mci}% chance of Mild Cognitive Impairment, and a {ad}% risk of Alzheimer Disease.",
        f"The neurocognitive risk stratification indicates {normal}% for normal function, {mci}% for Mild Cognitive Impairment, and {ad}% for Alzheimer Disease.",
        f"Probabilistic modeling of the patient data yields {normal}% for typical cognitive health, {mci}% for Mild Cognitive Impairment, and {ad}% for Alzheimer Disease.",
        f"Our predictive algorithms estimate a {normal}% chance of normal brain function, a {mci}% risk of Mild Cognitive Impairment, and a {ad}% probability of Alzheimer Disease.",
        f"The probability of normal cognitive function stands at {normal}%, while the likelihood of Mild Cognitive Impairment is estimated at {mci}%, and the risk of Alzheimer Disease is calculated to be {ad}%.",
        f"Current data suggests a {normal}% chance of normal cognition, a {mci}% probability of Mild Cognitive Impairment, and a {ad}% risk of Alzheimer Disease.",
        f"The patient cognitive profile indicates {normal}% likelihood of normal function, {mci}% possibility of Mild Cognitive Impairment, and {ad}% potential for Alzheimer Disease.",
        f"Our analysis points to a {normal}% probability of cognitive normalcy, {mci}% risk of Mild Cognitive Impairment, and {ad}% chance of Alzheimer Disease.",
        f"The data indicates {normal}% normal cognitive function, {mci}% risk of Mild Cognitive Impairment, and {ad}% likelihood of Alzheimer Disease.",
        f"Based on our comprehensive assessment, we estimate {normal}% probability of normal cognition, {mci}% chance of Mild Cognitive Impairment, and {ad}% risk of Alzheimer Disease.",
        f"The patient results suggest {normal}% normal cognitive health, {mci}% potential for Mild Cognitive Impairment, and {ad}% risk factor for Alzheimer Disease."
    ]

    interpretation_styles = [
        lambda s: f"Clinical Interpretation:\n{s}",
        lambda s: f"Neurocognitive Insights:\n{s}",
        lambda s: f"Expert Analysis:\n{s}",
        lambda s: f"Cognitive Health Evaluation:\n{s}",
        lambda s: f"Professional Assessment:\n{s}"
        lambda s: f"Diagnostic Perspective:\n{s}",
        lambda s: f"Cognitive Analysis:\n{s}",
        lambda s: f"Specialist Evaluation:\n{s}",
        lambda s: f"Neurological Review:\n{s}",
        lambda s: f"Clinical Assessment:\n{s}"
    ]

    interpretation_sentences = {
        "low": [
            "These findings are generally reassuring, indicating a predominantly healthy cognitive profile.",
            "The results suggest a relatively low risk of significant cognitive impairment at this time.",
            "Overall, the data points towards a favorable cognitive health status, with minimal concerns raised.",
            "The cognitive profile appears robust, with a strong indication of normal brain function.",
            "These results are encouraging, suggesting a high likelihood of maintained cognitive abilities.",
            "The patient cognitive health appears to be in good standing based on these indicators.",
            "This profile is consistent with normal age-related cognitive function, without significant red flags.",
            "The data suggests a strong cognitive reserve, which is associated with better brain health outcomes.",
            "These findings align with what we typically see in individuals with well-preserved cognitive function."
            "The presented data paints an optimistic picture of the patient cognitive health, suggesting robust neurological function.",
            "These findings are indicative of a well-preserved cognitive infrastructure, with minimal indications of neurodegeneration.",
            "The patient neurocognitive profile aligns closely with parameters typical of healthy brain aging, presenting no significant red flags.",
            "Analysis reveals a formidable cognitive reserve, which is associated with enhanced resilience against age-related cognitive decline.",
            "The data suggests a negligible risk of imminent cognitive deterioration, reflecting a healthy neural network.",
            "These results are congruent with optimal brain health, indicating efficient neural processing and connectivity.",
            "The patient cognitive landscape appears to be thriving, with strong indications of neuroplasticity and cognitive adaptability."
        ],
        "moderate": [
            "While not alarmingly aberrant, the findings do warrant vigilance and proactive cognitive health management.",
            "The results indicate a moderate level of cognitive vulnerability that necessitates careful monitoring and potential intervention.",
            "This cognitive profile suggests subtle neurological changes that, while not severe, merit attentive tracking and preventive measures.",
            "The data points to a state of cognitive flux, potentially indicative of early-stage neurodegenerative processes requiring close observation.",
            "These findings fall within a gray area, suggesting a need for more frequent cognitive assessments and potential lifestyle modifications.",
            "The results hint at incipient cognitive changes that, while not definitively pathological, call for heightened awareness and preventive strategies.",
            "This profile indicates a delicate balance in cognitive health, necessitating a proactive approach to maintain neural integrity.",
            "While not alarming, these results do warrant attention and further investigation.",
            "The findings indicate a moderate level of concern regarding the patient cognitive health.",
            "This cognitive profile suggests a need for vigilance and proactive measures to monitor cognitive function.",
            "The results point to potential early signs of cognitive change that require closer monitoring.",
            "While not definitive, these findings suggest an increased risk for future cognitive decline.",
            "The cognitive profile indicates a need for preventive strategies to maintain current function.",
            "These results fall in a gray area, necessitating more frequent cognitive assessments.",
            "The data suggests subtle cognitive changes that, while not severe, should be tracked carefully.",
            "This profile indicates a need for lifestyle modifications to potentially slow cognitive aging."
        ],
        "high": [
            "The findings raise significant concerns about the patient cognitive trajectory, indicating a high likelihood of active neurodegeneration.",
            "This cognitive profile is indicative of substantial neural compromise, necessitating immediate and comprehensive intervention.",
            "The data strongly suggests an accelerated neurodegenerative process that demands urgent medical attention and specialized care.",
            "These results align with patterns typically observed in advanced cognitive decline, calling for aggressive therapeutic strategies.",
            "The findings point to critical vulnerabilities in cognitive function, warranting immediate implementation of neuroprotective measures.",
            "This profile reveals alarming indicators of cognitive fragility, necessitating swift and multifaceted therapeutic intervention.",
            "The data paints a concerning picture of rapid cognitive deterioration, requiring immediate, intensive neurological care and support.",
            "These results raise significant concerns about the patient cognitive health and require immediate attention.",
            "The findings indicate a high risk of cognitive impairment that necessitates urgent and comprehensive evaluation.",
            "This cognitive profile is worrisome and calls for immediate intervention and thorough assessment.",
            "The data strongly suggests an active neurodegenerative process that requires prompt medical intervention.",
            "These results are consistent with significant cognitive decline, necessitating immediate specialized care.",
            "The findings point to a high likelihood of progressing cognitive impairment requiring urgent action.",
            "This profile indicates a critical need for comprehensive neurological and cognitive assessment.",
            "The data suggests advanced cognitive changes that warrant immediate therapeutic intervention.",
            "These results are alarming and indicative of significant cognitive vulnerability requiring swift action."
        ]
    }

    recommendation_styles = [
        lambda s: f"Recommended Action Plan:\n{s}",
        lambda s: f"Therapeutic Strategies:\n{s}",
        lambda s: f"Cognitive Health Roadmap:\n{s}",
        lambda s: f"Neurocognitive Management Plan:\n{s}",
        lambda s: f"Brain Health Optimization Strategy:\n{s}"
        lambda s: f"Suggested Course of Action:\n{s}",
        lambda s: f"Therapeutic Approach:\n{s}",
        lambda s: f"Cognitive Wellness Pathway:\n{s}",
        lambda s: f"Neurocognitive Care Plan:\n{s}",
        lambda s: f"Brain Wellness Enhancement Blueprint:\n{s}"

    ]

    recommendation_sentences = {
        "low": [
            "Implement a regimen of cognitive enhancement exercises to further bolster neural plasticity.",
            "Engage in regular cardiovascular exercise to promote cerebral blood flow and neurogenesis.",
            "Adopt a Mediterranean-style diet rich in omega-3 fatty acids and antioxidants to support brain health.",
            "Prioritize quality sleep hygiene to facilitate optimal cognitive recovery and consolidation.",
            "Participate in socially engaging activities to stimulate cognitive reserve and emotional well-being.",
            "Consider enrolling in a cognitive baseline study for long-term monitoring of neural health.",
            "Explore mindfulness meditation practices to enhance cognitive flexibility and stress resilience.",
            "Engage in multilingual activities or learn a new language to promote cognitive flexibility.",
            "Regularly challenge spatial reasoning skills through complex puzzles or 3D modeling activities.",
            "Incorporate intermittent fasting regimens to potentially enhance neuroprotective mechanisms.",
            "Consider a baseline Blood Biomarker Test to establish cognitive health markers for future reference.",
            "Maintain regular cognitive health check-ups with the primary care physician.",
            "Encourage engagement in cognitively stimulating activities as part of a brain-healthy lifestyle.",
            "Discuss family history and potential lifestyle modifications with a healthcare provider.",
            "Stay informed about the latest research in cognitive health and prevention strategies.",
            "Engage in regular physical exercise, which has been shown to support brain health.",
            "Maintain a balanced diet rich in omega-3 fatty acids, antioxidants, and vitamins.",
            "Prioritize quality sleep, aiming for 7-9 hours per night.",
            "Practice stress-reduction techniques such as meditation or mindfulness.",
            "Stay socially active and engaged in community activities.",
            "Consider learning a new language or musical instrument to promote cognitive flexibility.",
            "Engage in puzzles, crosswords, or strategy games to challenge cognitive functions.",
            "Maintain regular vision and hearing check-ups to support overall cognitive health.",
            "Consider participating in a cognitive baseline study for long-term monitoring.",
            "Discuss the potential benefits of cognitive training programs with a specialist."
        ],
        "moderate": [
        "Undergo comprehensive neuropsychological testing and a Blood Biomarker Test to pinpoint specific cognitive domains requiring attention.",
        "Consult with a cognitive neurologist to develop a tailored cognitive preservation strategy, including regular Blood Biomarker Tests.",
        "Explore cognitive rehabilitation programs designed to target areas of emerging weakness, complemented by periodic Blood Biomarker Tests.",
        "Consider transcranial magnetic stimulation (TMS) therapy to potentially enhance neural function, monitored through Blood Biomarker Tests.",
        "Investigate the benefits of neurofeedback training for cognitive performance optimization, alongside regular Blood Biomarker assessments.",
        "Enroll in a structured cognitive training program utilizing adaptive difficulty algorithms, with Blood Biomarker Tests to track progress.",
        "Explore the potential neuroprotective benefits of certain nutraceuticals under medical supervision, monitored through Blood Biomarker Tests.",
        "Engage in dual-task training exercises to enhance cognitive flexibility and processing speed, with Blood Biomarker Tests to measure effectiveness.",
        "Participate in virtual reality-based cognitive training sessions for immersive brain stimulation, complemented by Blood Biomarker monitoring.",
        "Consider the adoption of a ketogenic diet under medical supervision for potential neuroprotective effects, tracked through Blood Biomarker Tests.",
        "Schedule regular Blood Biomarker Tests to evaluate specific markers associated with cognitive health.",
        "Undergo a comprehensive neuropsychological assessment and Blood Biomarker Test to evaluate various cognitive domains.",
        "Consult with a neurologist specializing in cognitive disorders for a more in-depth evaluation, including Blood Biomarker analysis.",
        "Consider participating in cognitive monitoring programs for ongoing assessment, incorporating regular Blood Biomarker Tests.",
        "Evaluate and potentially modify lifestyle factors that may impact cognitive health, monitored through Blood Biomarker Tests.",
        "Discuss the potential benefits of cognitive enhancement techniques with a specialist, using Blood Biomarker Tests to gauge effectiveness.",
        "Explore memory aids and organizational tools to support daily cognitive function, with Blood Biomarker Tests to track overall brain health.",
        "Consider enrolling in a cognitive rehabilitation program, with progress monitored through regular Blood Biomarker Tests.",
        "Discuss the potential use of cholinesterase inhibitors with a neurologist, alongside Blood Biomarker monitoring.",
        "Evaluate cardiovascular health, as it closely linked to cognitive function, through both cardiac and cognitive Blood Biomarker Tests.",
        "Consider participating in clinical trials focused on early intervention in cognitive decline, which often include Blood Biomarker assessments.",
        "Explore the potential benefits of transcranial magnetic stimulation (TMS) therapy, with Blood Biomarker Tests to measure its impact.",
        "Discuss the role of advanced neuroimaging techniques and Blood Biomarker Tests in monitoring cognitive changes.",
        "Consider genetic counseling to understand hereditary risk factors, complemented by Blood Biomarker analysis.",
        "Explore mindfulness-based stress reduction programs to support cognitive health, with Blood Biomarker Tests to quantify benefits."
        ],
        "high": [
        "Immediately consult with a neurologist specializing in neurodegenerative disorders for comprehensive evaluation, including urgent Blood Biomarker Tests.",
        "Explore eligibility for cutting-edge clinical trials targeting cognitive decline and neuroprotection, which typically involve extensive Blood Biomarker analysis.",
        "Investigate the potential benefits of deep brain stimulation (DBS) in managing cognitive symptoms, monitored through regular Blood Biomarker Tests.",
        "Consider the implementation of intensive cognitive rehabilitation protocols, with progress tracked via Blood Biomarker Tests.",
        "Explore emerging pharmacological interventions targeting neural repair and regeneration, with effectiveness measured through Blood Biomarker analysis.",
        "Engage in daily cognitive training sessions utilizing AI-driven adaptive programs, complemented by periodic Blood Biomarker assessments.",
        "Implement a rigorous physical therapy regimen designed to maintain neural-muscular connections, monitored through specialized Blood Biomarker Tests.",
        "Explore the potential of hyperbaric oxygen therapy for cognitive support, with effects measured via Blood Biomarker Tests.",
        "Consider the adoption of a medically supervised, neuroprotective ketogenic diet, with metabolic and cognitive impacts tracked through Blood Biomarker Tests.",
        "Investigate the potential benefits of stem cell therapies for neural regeneration, monitored closely with Blood Biomarker Tests.",
        "Urgently schedule comprehensive Blood Biomarker Tests to assess markers specific to cognitive decline and Alzheimer Disease.",
        "Undergo an immediate and comprehensive neurological examination, including detailed cognitive testing and extensive Blood Biomarker analysis.",
        "Arrange for a consultation with a neurologist specializing in neurodegenerative disorders, who will likely order specialized Blood Biomarker Tests.",
        "Consider additional advanced neuroimaging studies as recommended by the specialist, coupled with targeted Blood Biomarker Tests.",
        "Evaluate the need for potential interventions or clinical trials targeting cognitive decline, which often involve regular Blood Biomarker monitoring.",
        "Discuss long-term care planning and support systems with healthcare providers and family members, informed by ongoing Blood Biomarker assessments.",
        "Investigate genetic counseling options for family members, given the elevated risk profile, supported by familial Blood Biomarker analysis.",
        "Explore the potential benefits of deep brain stimulation (DBS) in managing cognitive decline, with effectiveness tracked through Blood Biomarker Tests.",
        "Consider participating in clinical trials for novel Alzheimer Disease treatments, which typically include extensive Blood Biomarker monitoring.",
        "Discuss the potential use of amyloid-targeting therapies with a specialist, guided by specific Blood Biomarker Tests.",
        "Evaluate the need for in-home safety modifications to support independent living, with cognitive status regularly assessed through Blood Biomarker Tests.",
        "Consider enrolling in a specialized day program for cognitive support, with progress monitored via regular Blood Biomarker Tests.",
        "Explore options for cognitive prosthetics and assistive technologies, with their impact measured through periodic Blood Biomarker assessments.",
        "Discuss the potential benefits of intensive cognitive rehabilitation programs, with effectiveness gauged through regular Blood Biomarker Tests.",
        "Consider the role of nutritional interventions specifically targeted at neurodegenerative processes, monitored through specialized Blood Biomarker Tests."
        ]
    }

    cognitive_exercise_styles = [
        lambda s: f"Recommended Cognitive Exercises:\n{s}",
        lambda s: f"Brain Training Activities:\n{s}",
        lambda s: f"Neurocognitive Enhancement Techniques:\n{s}",
        lambda s: f"Mental Fitness Regimen:\n{s}",
        lambda s: f"Cognitive Stimulation Exercises:\n{s}"
    ]

    cognitive_exercises = [
        "Engage in complex, multi-step cooking recipes to challenge executive function and memory.",
        "Practice the ancient art of memory palaces to enhance spatial memory and recall.",
        "Participate in improvisational theater to boost cognitive flexibility and social cognition.",
        "Engage in bird watching, honing observational skills and attention to detail.",
        "Learn to juggle, promoting hand-eye coordination and spatial awareness.",
        "Practice mindfulness meditation to enhance attention and emotional regulation.",
        "Engage in speedcubing (rapid Rubik cube solving) to boost problem-solving skills.",
        "Participate in orienteering to challenge spatial navigation and decision-making.",
        "Learn calligraphy to enhance fine motor skills and attention to detail.",
        "Engage in competitive memory sports to push the limits of mnemonic abilities.",
        "Practice mental arithmetic using the abacus method to enhance numerical processing.",
        "Participate in debate clubs to sharpen critical thinking and verbal reasoning.",
        "Learn American Sign Language to boost spatial-visual processing and memory.",
        "Engage in blindfolded chess to challenge visualization and strategic planning.",
        "Practice daily journaling with your non-dominant hand to stimulate neural pathways.",
        "Engage in complex origami projects to enhance spatial reasoning and fine motor skills.",
        "Participate in escape room challenges to boost problem-solving under pressure.",
        "Learn to read and compose music to enhance auditory processing and pattern recognition.",
        "Engage in regular sessions of Tai Chi to improve bodily-kinesthetic awareness.",
        "Practice speed reading techniques to enhance visual processing and comprehension.",
        "Sudoku puzzles of increasing difficulty.",
        "Crossword puzzles from various sources.",
        "Lumosity brain training games.",
        "Chess or other strategic board games.",
        "Learning a new language using apps like Duolingo.",
        "Memorization exercises, such as poetry or speeches.",
        "Mental math calculations without a calculator.",
        "Jigsaw puzzles of varying complexity.",
        "Reading and discussing complex literature.",
        "Writing creative fiction or non-fiction pieces.",
        "Playing musical instruments or learning music theory.",
        "Engaging in debate or public speaking activities.",
        "Solving riddles and logic puzzles.",
        "Participating in quiz bowls or trivia nights.",
        "Learning new dance routines.",
        "Practicing mindfulness and meditation.",
        "Engaging in art activities like painting or sculpting.",
        "Playing video games designed for cognitive enhancement.",
        "Participating in escape room challenges.",
        "Studying a new academic subject."
    ]

    limitation_styles = [
        lambda s: f"Limitations of Analysis:\n{s}",
        lambda s: f"Caveats and Considerations:\n{s}",
        lambda s: f"Interpretative Boundaries:\n{s}",
        lambda s: f"Analytical Constraints:\n{s}",
        lambda s: f"Data Interpretation Caveats:\n{s}"
    ]

    limitation_sentences = [
        "It is imperative to recognize that this analysis, while comprehensive, is constrained by the specific modalities of data collection employed.",
        "The predictive power of this assessment is bounded by the current state of neurocognitive science and may evolve with future discoveries.",
        "This evaluation provides a snapshot of cognitive health and may not capture the dynamic nature of neural function over time.",
        "The interpretation of these results should be tempered by an understanding of the complex interplay between genetics, environment, and lifestyle factors.",
        "While highly informative, this analysis does not account for potential epigenetic factors that may influence cognitive trajectories.",
        "The accuracy of these findings may be influenced by factors not captured in the current dataset, such as recent life events or acute health changes.",
        "This assessment, while detailed, cannot fully encapsulate the nuanced presentation of cognitive function in real-world scenarios.",
        "The extrapolation of these results to long-term cognitive outcomes should be approached with caution, given the multifaceted nature of brain health.",
        "It is crucial to consider that cognitive reserve and neuroplasticity may modulate the manifestation of the risk factors identified in this analysis.",
        "The interpretation of these findings should be contextualized within the broader landscape of the patient overall health and life circumstances.",
        "It is crucial to note that this analysis is based solely on genetic testing, MRI scans, and health records, which provide a limited view of overall cognitive health.",
        "While informative, this assessment is not definitive and should be considered in conjunction with a comprehensive clinical evaluation.",
        "The limitations of this analysis should be recognized, as it does not account for all factors that may influence cognitive health and function.",
        "This report provides a snapshot based on available data and does not capture the dynamic nature of cognitive health.",
        "The predictive value of these findings may be limited by factors not captured in the current dataset.",
        "It important to consider that cognitive health is influenced by numerous factors beyond what measured in this analysis.",
        "This assessment does not account for recent lifestyle changes or acute health events that may impact cognitive function.",
        "The interpretation of these results may evolve as our understanding of cognitive health biomarkers advances.",
        "This analysis does not capture the nuanced presentation of cognitive symptoms in a clinical setting.",
        "The predictive accuracy of these findings may vary based on individual genetic and environmental factors."
    ]

    followup_styles = [
        lambda s: f"Follow-up Recommendations:\n{s}",
        lambda s: f"Ongoing Monitoring Plan:\n{s}",
        lambda s: f"Longitudinal Assessment Strategy:\n{s}",
        lambda s: f"Future Evaluation Roadmap:\n{s}"
    ]

    followup_sentences = [
        "Regular follow-up assessments are advised to monitor any changes in cognitive status over time.",
        "Periodic re-evaluation is recommended to track cognitive health trajectories and adjust interventions as needed.",
        "Ongoing monitoring of cognitive function is crucial for early detection of any significant changes.",
        "We suggest scheduling a follow-up cognitive assessment in 6-12 months to track any potential changes.",
        "Continuous engagement with healthcare providers is key to managing and monitoring cognitive health.",
        "Future assessments should include a broader range of cognitive tests to provide a more comprehensive picture.",
        "We recommend maintaining a cognitive health diary to track subjective changes between formal assessments.",
        "Regular check-ins with a cognitive health specialist can help fine-tune management strategies over time.",
        "Participation in longitudinal cognitive health studies could provide valuable insights into individual cognitive trajectories.",
        "Periodic reassessment of biomarkers and neuroimaging can help track the progression or stability of cognitive health."
    ]

    report = random_sentences(intro_sentences, 1) + " " + random_sentences(probability_sentences, 1) + " "
    report += random_sentences(interpretation_sentences[risk_level], 2) + " "
    report += random_sentences(recommendation_sentences[risk_level], 2) + " "
    report += random_sentences(limitation_sentences, 1) + " " + random_sentences(followup_sentences, 1) + " "

    # report += f"Patient ID: {fake.uuid4()}\n"
    # report += f"Assessment Date: {fake.date_this_year()}\n"
    # report += f"Evaluating Clinician: {random.choice(['Dr.', 'Prof.', 'Neurologist'])} {fake.name()}, {random.choice(['MD', 'PhD', 'MD-PhD'])}"

    return report

data = []
for _ in range(18000):
    normal, mci, ad = generate_probabilities()
    input_text = f"Cognitive Normal: {normal}%\nMild Cognitive Impairment: {mci}%\nAlzheimer Disease: {ad}%"
    output = generate_medical_report(normal, mci, ad)
    data.append({
        "Instruction": "Make a medical report with a summary",
        "Input": input_text,
        "Output": output
    })

df = pd.DataFrame(data)
df.to_csv("ADMR.csv", index=False)
print("Dataset generated and saved to 'ADMR.csv'")