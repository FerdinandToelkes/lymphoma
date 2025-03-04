from pamly import Diagnosis

"""
This file contains the mapping of the diagnosis labels to integers and vice versa,
such that it is uniform across the entire project.
"""

LABELS_MAP_STRING_TO_INT = {
            "Unknown": int(Diagnosis("Unknown")),       # for unknown diagnosis
            "HL":      int(Diagnosis("HL")),            # Hodgkin Lymphoma
            "DLBCL":   int(Diagnosis("DLBCL")),         # Diffuse Large B-Cell Lymphoma
            "CLL":     int(Diagnosis("CLL")),           # Chronic Lymphocytic Leukemia
            "FL":      int(Diagnosis("FL")),            # Follicular Lymphoma
            "MCL":     int(Diagnosis("MCL")),           # Mantle Cell Lymphoma
            "LYM":     int(Diagnosis("LTDS")),          # Lymphadenitis
            "LTS":     int(Diagnosis("LTDS")),          # Lymphadenitis
            "LTDS":    int(Diagnosis("LTDS")),          # Lymphadenitis
        }

LABELS_MAP_INT_TO_STRING= {
            int(Diagnosis("Unknown")): "Unknown",      # for unknown diagnosis
            int(Diagnosis("HL")):      "HL",           # Hodgkin Lymphoma
            int(Diagnosis("DLBCL")):   "DLBCL",        # Diffuse Large B-Cell Lymphoma
            int(Diagnosis("CLL")):     "CLL",          # Chronic Lymphocytic Leukemia
            int(Diagnosis("FL")):      "FL",           # Follicular Lymphoma
            int(Diagnosis("MCL")):     "MCL",          # Mantle Cell Lymphoma
            int(Diagnosis("LTDS")):    "LTDS",         # Lymphadenitis
        }

if __name__ == "__main__":
    print(LABELS_MAP_STRING_TO_INT)
    print(LABELS_MAP_INT_TO_STRING)