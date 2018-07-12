
import pandas as pd

# READ FILE
tehsil_sch = pd.read_excel("Tehsil Schools.xlsx")

print("Shape of the dataset is : {}\n".format(tehsil_sch.shape))

#FIND THE COLUMN NAMES TO USE TO ANSWER QUESTIONS
print(tehsil_sch.columns.values)

# QUESTION 1
print("="*60)
list_of_schools_q1 = tehsil_sch[tehsil_sch["No. of Students appeared in 10th, BISE Exam:2014"] == 0]
print(list_of_schools_q1["Name of School"],"\n")

print("="*60)
# QUESTION 2
list_of_schools_q2 = tehsil_sch[tehsil_sch["No. of Students  in 9th (2012) as per Registeration"] > 50]

print("Large Sizes Schools are {} % of Total Schools\n".format(round(len(list_of_schools_q2) * 100 / len(tehsil_sch),2)))

print("="*60)
# QUESTION 3
max_drop_out = round(list_of_schools_q2["% dropout"].max(),2)
school_name = list_of_schools_q2["Name of School"][list_of_schools_q2["% dropout"].argmax()]
print(school_name,"School has the highest % dropout({}%)".format(max_drop_out))