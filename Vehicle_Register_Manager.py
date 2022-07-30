import csv
from datetime import datetime

def vehicle_entry(vehicle_number):

    print("Updating Vehicle Register : " + vehicle_number)
    # enter vehile number, date, time in register
    text_file = open("vehichle_register.txt", "a")
    text_file.write(vehicle_number)
    text_file.write(", ")
    text_file.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    text_file.write("\n")
    text_file.close()

def validate_vehicle(vehicle_number):
    print("Validating vehicle : " + vehicle_number)
    # reading csv file
    with open("Records.txt", 'r') as csvfile:

        # creating a csv reader object
        csvreader = csv.reader(csvfile)
        # including only first column consisting of numberplates
        included_cols = [0]
        authorization_status = False
        for row in csvreader:
            content = list(row[i] for i in included_cols)
            # Checking if detected numberplate exists in the records
            if (vehicle_number in content):
                authorization_status = True
                break;

        if(authorization_status):
            print("Authorized Vehicle")
        else:
            print("Unauthorized vehicle")

