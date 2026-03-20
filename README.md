# car-evaluation-project
## Problem
โปรเจคนี้ทำขึ้นเพื่อทำนายคุณภาพของรถยนต์จากข้อมูลคุณสมบัติต่าง ๆ เช่น ราคา ค่าบำรุงรักษา ความปลอดภัย และจำนวนที่นั่ง โดยใช้ Machine Learning เข้ามาช่วยในการจำแนกประเภทของรถ

## Dataset
ใช้ Car Evaluation Dataset ซึ่งเป็นข้อมูลแบบ categorical ทั้งหมด ประกอบไปด้วย feature ดังนี้
- buying: ราคาซื้อ
- maint: ค่าบำรุงรักษา
- doors: จำนวนประตู
- persons: จำนวนที่นั่ง
- lug_boot: ขนาดที่เก็บของ
- safety: ความปลอดภัย
- class: ประเภทของรถ (ตัวแปรเป้าหมาย)

## Data Analysis
จากการสำรวจข้อมูลเบื้องต้น พบว่า
- ข้อมูลไม่มี missing values
- ทุก feature เป็น categorical
- class มีหลายระดับ เช่น unacc, acc, good และ vgood

## Model
ในโปรเจคนี้เลือกใช้ RandomForestClassifier เนื่องจากสามารถจัดการกับข้อมูลประเภท categorical ได้ดี  
มีการแบ่งข้อมูลเป็น train/test และมีการทำ cross validation  
นอกจากนี้ยังมีการปรับค่า hyperparameter ด้วย GridSearchCV เพื่อให้ได้ผลลัพธ์ที่ดีขึ้น

## Result
ประเมินผลด้วย classification report และ confusion matrix  
ผลลัพธ์ที่ได้อยู่ในระดับที่ดี และ model สามารถทำนายได้ถูกต้องในหลายกรณี

## Deployment
นำโมเดลไปใช้งานผ่าน Streamlit โดยให้ผู้ใช้เลือกค่าของ feature ต่าง ๆ และระบบจะแสดงผลการทำนายทันที

## How to run
1. ติดตั้ง library
2. รันคำสั่ง streamlit run app.py
