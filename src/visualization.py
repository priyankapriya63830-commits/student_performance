import matplotlib
matplotlib.use('Agg')  # no GUI windows

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# Create output folder
# --------------------------------------------------
output_dir = os.path.join("artifacts", "plots")
os.makedirs(output_dir, exist_ok=True)

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
data = pd.read_csv(r"D:\priya_internship\data\StudentPerformanceFactors.csv")

# Clean data
data = data.drop_duplicates()
data = data.dropna()

# --------------------------------------------------
# 1. Motivation Level Distribution (Countplot)
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

sns.countplot(
    data=data,
    x='Motivation_Level',
    order=data['Motivation_Level'].value_counts().index,
    ax=ax
)

ax.set_title("Distribution of Motivation Levels")

fig.savefig(os.path.join(output_dir, "01_motivation_distribution.png"),
            dpi=300, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------
# 2. Box Plot – Gender vs Exam Score
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

sns.boxplot(data=data, x='Gender', y='Exam_Score', ax=ax)

ax.set_title("Exam Score by Gender")

fig.savefig(os.path.join(output_dir, "02_gender_boxplot.png"),
            dpi=300, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------
# 3. Violin Plot – School Type
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

sns.violinplot(data=data, x='School_Type', y='Exam_Score', ax=ax)

ax.set_title("Exam Score by School Type")

fig.savefig(os.path.join(output_dir, "03_schooltype_violin.png"),
            dpi=300, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------
# 4. Scatter Plot – Hours vs Score
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

sns.scatterplot(
    data=data,
    x='Hours_Studied',
    y='Exam_Score',
    hue='Gender',
    ax=ax
)

ax.set_title("Hours Studied vs Exam Score")

fig.savefig(os.path.join(output_dir, "04_scatter_hours_score.png"),
            dpi=300, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------
# 5. Bar Plot – Motivation vs Score
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

sns.barplot(
    data=data,
    x='Motivation_Level',
    y='Exam_Score',
    ax=ax
)

ax.set_title("Avg Exam Score by Motivation")

fig.savefig(os.path.join(output_dir, "05_motivation_bar.png"),
            dpi=300, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------
# 6. Count Plot – Family Income
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

sns.countplot(
    data=data,
    x='Family_Income',
    ax=ax
)

ax.set_title("Family Income Distribution")

fig.savefig(
    os.path.join(output_dir, "06_family_income.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

# --------------------------------------------------
# 7. Heatmap – Correlation
# --------------------------------------------------
corr = data[
    [
        'Hours_Studied',
        'Attendance',
        'Sleep_Hours',
        'Previous_Scores',
        'Tutoring_Sessions',
        'Physical_Activity',
        'Exam_Score'
    ]
].corr()

fig, ax = plt.subplots(figsize=(10, 6))

sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    ax=ax
)

ax.set_title("Correlation Heatmap")

fig.savefig(
    os.path.join(output_dir, "07_heatmap.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)
# --------------------------------------------------
# 8. Line Plot – Study Hours vs Score
# --------------------------------------------------
avg_score = data.groupby('Hours_Studied')['Exam_Score'].mean().reset_index()

fig, ax = plt.subplots(figsize=(8, 5))

sns.lineplot(data=avg_score,
             x='Hours_Studied',
             y='Exam_Score',
             ax=ax)

ax.set_title("Study Hours vs Average Score")

fig.savefig(os.path.join(output_dir, "08_lineplot.png"),
            dpi=300, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------
# 9. Pair Plot
# --------------------------------------------------
pair = sns.pairplot(
    data[
        [
            'Hours_Studied',
            'Attendance',
            'Sleep_Hours',
            'Previous_Scores',
            'Exam_Score'
        ]
    ]
)

pair.fig.savefig(
    os.path.join(output_dir, "09_pairplot.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close('all')

# --------------------------------------------------
# 10. Strip Plot – Parental Involvement
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

sns.stripplot(
    data=data,
    x='Parental_Involvement',
    y='Exam_Score',
    jitter=True,
    ax=ax
)

ax.set_title("Exam Score by Parental Involvement")

fig.savefig(os.path.join(output_dir, "10_parental_involvement.png"),
            dpi=300, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------
print(f"All plots saved successfully in: {output_dir}")