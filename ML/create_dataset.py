import os
import shutil

# set a destination folder for saving the seperated images
dest_add = r"F:\Code practice data 2"

Base_folder_name = r"Data_set"

Base_folder_add = os.path.join(dest_add, Base_folder_name)

if not os.path.exists(Base_folder_add):
    os.makedirs(Base_folder_add)

# Link the folder you dowloaded from drive
source = r"C:\Users\786 COMPUTERS\Downloads\PH2 Dataset images-20231103T210031Z-001\PH2 Dataset images"

Atypical = {'address': "Atypical Nevus", 'index': [2, 4, 13, 15, 19, 21, 27, 30, 32, 33, 37, 40, 43, 47, 48, 49, 57, 75, 76, 78, 120, 126, 137, 138, 139, 149, 153, 157, 164, 166, 169, 171, 210, 347, 155, 376,
                                                   6, 8, 14, 18, 23, 31, 36, 154, 170, 226, 243, 251, 254, 256, 278, 279, 280, 304, 305, 306, 312, 328, 331, 339, 356, 360, 368, 369, 370, 382, 386, 388, 393, 396, 398, 427, 430, 431, 432, 433, 434, 436, 437]}
Common = {'address': "Common Nevus", 'index': [3, 9, 16, 22, 24, 25, 35, 38, 42, 44, 45, 50, 92, 101, 103, 112, 118, 125, 132, 134, 135, 144, 146, 147, 150, 152, 156, 159, 161, 162, 175, 177, 162, 198, 200, 10, 17,
                                               20, 39, 41, 105, 107, 108, 133, 142, 143, 160, 173, 176, 196, 197, 199, 203, 204, 206, 207, 208, 364, 365, 367, 371, 372, 375, 374, 378, 379, 380, 381, 383, 385, 384, 389, 390, 392, 394, 395, 397, 399, 400, 402]}
Mela = {'address': "Melanoma", 'index': [58, 61, 63, 64, 65, 80, 85, 88, 90, 91, 168, 211, 219, 240, 242, 284, 285,
                                         348, 349, 403, 404, 405, 407, 408, 409, 410, 413, 417, 418, 419, 406, 411, 420, 421, 423, 424, 425, 426, 429, 435]}

classes = [Atypical, Common, Mela]

for i in classes:
    sub_fol = os.path.join(Base_folder_add, i['address'])
    # Ensure the subfolder exists or create it
    if not os.path.exists(sub_fol):
        os.makedirs(sub_fol, exist_ok=True)
    counter = 0
    for j in i['index']:

        # IF THE IMAGE NO. IS IN 0-9
        a = "IMD00"+str(j)
        b = "IMD00"+str(j)+"_Dermoscopic_Image"
        file = "IMD00" + str(j) + ".BMP"
        from_add = os.path.join(source, a, b, file)

        if os.path.isfile(from_add):
            print(f"File {a}.BMP copied to {i['address']}")
            shutil.copyfile(from_add, os.path.join(
                sub_fol, f'image{counter}.jpg'))
            counter += 1

        # IF THE IMAGE NO. IS IN 10-99
        a = "IMD0"+str(j)
        b = "IMD0"+str(j)+"_Dermoscopic_Image"
        file = "IMD0" + str(j) + ".BMP"
        from_add = os.path.join(source, a, b, file)

        if os.path.isfile(from_add):
            print(f"File {a}.BMP copied to {i['address']}")
            shutil.copyfile(from_add, os.path.join(
                sub_fol, f'image{counter}.jpg'))
            counter += 1

        # IF THE IMAGE NO. IS IN 100-999
        a = "IMD"+str(j)
        b = "IMD"+str(j)+"_Dermoscopic_Image"
        file = "IMD" + str(j) + ".BMP"
        from_add = os.path.join(source, a, b, file)

        if os.path.isfile(from_add):
            print(f"File {a}.BMP copied to {i['address']}")
            shutil.copyfile(from_add, os.path.join(
                sub_fol, f'image{counter}.jpg'))
            counter += 1
