'''
Classification result report.

Dominic Spata,
Real-Time Computer Vision,
Institut fuer Neuroinformatik,
Ruhr University Bochum.
'''


import numpy as np
import copy


LABEL_NAMES = [
    'Speed limit 20',
    'Speed limit 30',
    'Speed limit 50',
    'Speed limit 60',
    'Speed limit 70',
    'Speed limit 80',
    'Derestrict 80',
    'Speed limit 100',
    'Speed limit 120',
    'Prohibit overtaking',
    'Prohibit overtaking (trucks)',
    'Right of way',
    'Priority street',
    'Yield way',
    'Stop',
    'No entry',
    'No entry (trucks)',
    'One way street',
    'Danger',
    'Attention road curves left',
    'Attention road curves right',
    'Attention S curve',
    'Attention bumpy road',
    'Attention slippery road',
    'Attention road will narrow',
    'Attention construction site',
    'Attention traffic lights',
    'Attention pedestrians',
    'Attention playing children',
    'Attention bicycle',
    'Attention snowfall',
    'Attention deer crossing',
    'Derestrict',
    'Turn right',
    'Turn left',
    'Forward',
    'Forward or right',
    'Forward or left',
    'Pass right',
    'Pass left',
    'Roundabout',
    'Derestrict overtaking',
    'Derestrict overtaking (trucks)'
]

LATEX_HEADER = [
    "\\documentclass{article}",
    "\\usepackage[paperwidth=250cm,paperheight=250cm]{geometry}",
    "\\usepackage{graphicx}",
    "\\usepackage[table]{xcolor}",
    "\\begin{document}"
]

LATEX_FOOTER = [
    "\\end{document}"
]


class Report:
    '''
    Class for calculating and holding a report on classification results.
    '''

    def __init__(self, labels, predicted_labels, label_names):
        '''
        Constructs the classification report.

        Arguments:
            labels              -- The vector of actual labels of the examples.
            predicted_labels    -- The vector of labels predicted by the model for the same examples.
            label_names         -- A list of textual descriptions of each possible class label.
        '''

        self.class_count = len(label_names)

        self.label_counts = np.zeros([self.class_count], dtype = np.int32)
        self.confusion_matrix_absolute = np.zeros([self.class_count, self.class_count], dtype = np.int32)
        self.correct_classifications = 0
        self.total_classifications = len(labels)

        for i in range(self.total_classifications):
            self.label_counts[labels[i]] += 1
            self.confusion_matrix_absolute[labels[i]][predicted_labels[i]] += 1

            if labels[i] == predicted_labels[i]:
                self.correct_classifications += 1
            
        self.confusion_matrix_relative = self.confusion_matrix_absolute / self.label_counts.reshape([self.class_count, 1])
        self.label_names = label_names

        self.error = 1 - self.correct_classifications / self.total_classifications

    def __sub__(self, other):
        self_copy = copy.deepcopy(self)

        self_copy.correct_classifications -= other.correct_classifications
        self_copy.error -= other.error
        self_copy.confusion_matrix_absolute -= other.confusion_matrix_absolute
        self_copy.confusion_matrix_relative -= other.confusion_matrix_relative

        return self_copy

    def get_sparse_representation(self, matrix, epsilon):
        sparse_representation = [list() for i in range(self.class_count)]

        for i in range(self.class_count):
            for j in range(self.class_count):
                if abs(matrix[i, j]) > epsilon:
                    sparse_representation[i].append((j, matrix[i, j]))

        return sparse_representation

    def get_sparse_string(self, matrix, epsilon):
        sparse_representation = self.get_sparse_representation(matrix, epsilon)

        string = str()

        for i in range(self.class_count):
            string += self.label_names[i] + ":\t" + str(matrix[i, i]) + "\n"

            for sparse_tuple in sparse_representation[i]:
                if sparse_tuple[0] == i: continue
                string += "\t" + self.label_names[sparse_tuple[0]] + ":\t" + str(sparse_tuple[1]) + "\n"

            string += "\n"

        return string

    def bold(self, text):
        '''
        Create bold text for the Latex document.

        Argument:
            text -- The text to set bold.
        
        Returns:
            The bold text.
        '''

        return "\\textbf{" + text + "}"

    def color(self, value):
        if value >= 0:
            return "green!" + str(value * 50)

        return "red!" + str(-value * 50)

    def begin_matrix(self, column_specifier):
        return "\\begin{tabular}{" + column_specifier + "}"

    def end_matrix(self):
        return "\\end{tabular}"

    def vector_to_line(self, vector, bold_index = -1, color = None):
        cells = [str(value) for value in vector]

        if bold_index != -1:
            cells[bold_index] = self.bold(cells[bold_index])

        if color is not None:
            for i in range(len(vector)):
                cells[i] = "\\cellcolor{" + color(vector[i]) + "}" + cells[i]

        return "&".join(cells) + "\\\\"

    def matrix_to_tabular(self, matrix, color = None):
        return [self.vector_to_line(matrix[i], color = color, bold_index = i) + "\\hline" for i in range(matrix.shape[0])]

    def matrix_column_headings(self, column_headings):
        return "&" + "&".join(["\\rotatebox{90}{" + self.bold(column_heading) + "}" for column_heading in column_headings]) + "\\\\\\hline\\hline"

    def matrix_row_headings(self, matrix_inner, row_headings):
        return [self.bold(row_headings[i]) + "&" + matrix_inner[i] for i in range(len(matrix_inner))]

    def matrix_to_table(self, matrix, row_headings, column_headings, color = None):
        table = list()

        table.append(self.begin_matrix("|r||" + "c|" * len(column_headings) + "|") + "\\hline")

        table.append(self.matrix_column_headings(column_headings))

        matrix_inner = self.matrix_to_tabular(matrix, color = color)
        table += self.matrix_row_headings(matrix_inner, row_headings)

        table.append(self.end_matrix())

        return table

    def dump(self, file):
        '''
        Dumps a Latex-based representation of the classification report to file.

        Arguments:
            file -- The file (with writing access) where to dump the report.
        '''

        lines = list()

        lines += LATEX_HEADER

        lines.append("\\section{Overview}")

        lines.append(self.begin_matrix("|l|l|") + "\\hline")

        lines.append(self.bold("Class count") + "&" + str(self.class_count) + "\\\\\\hline")
        lines.append(self.bold("Correct classifications") + "&" + str(self.correct_classifications) + "\\\\\\hline")
        lines.append(self.bold("Total classifications") + "&" + str(self.total_classifications) + "\\\\\\hline")
        lines.append(self.bold("Error") + "&" + str(self.error) + "\\\\\\hline")

        lines.append(self.end_matrix())

        lines.append("")
        lines.append("\\section{Confusion matrix (absolute)}")
        lines += self.matrix_to_table(self.confusion_matrix_absolute, self.label_names, self.label_names)

        lines.append("")
        lines.append("\\section{Confusion matrix (relative)}")
        lines += self.matrix_to_table(self.confusion_matrix_relative, self.label_names, self.label_names, color = self.color)

        lines += LATEX_FOOTER

        file.write("\n".join(lines))