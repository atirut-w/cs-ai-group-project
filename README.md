# CS AI Group Project
งานโปรเจคที่ 1 ของวิชา Fundamentals of Artificial Intelligence

### รายระเอียดโปรเจค
สร้าง regression models สำหรับการสร้าง AI model สำหรบการทำนายราคาของบ้านในตลาดอสังหาริมทรัพย์

dataset ที่ใช้: [Housing Price Prediction](https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction/data)

มี Features ของข้อมูลทั้งหมด 13 features ได้แก่
- Price: ราคาของบ้าน
- Area: พื้นที่ทั้งหมดของบ้านในหน่วยตารางฟุต
- Bedrooms: จำนวนห้องนอน
- Bathrooms: จำนวนห้องน้ำ
- Stories: จำนวนชั้น.
- Mainroad: บ้านติดอยู่กับถนนหลัก
- Guestroom: มีห้องพักแขก
- Basement: ชั้นใต้ดิน
- Hot water heating: มีระบบทำน้ำอุ่น
- Airconditioning: มีเครื่องปรับอากาศ
- Parking: จำนวนพื้นที่จอดรถในบ้าน
- Prefarea: บ้านอยู่ในพื้นที่ต้องการ
- Furnishing status: สถานะการตกแต่งในบ้าน

---

### ขั้นตอนการตัว models ไปพัฒนาต่อ

1. clone ตัวโปรเจค
```
git clone https://github.com/atirut-w/cs-ai-group-project.git
```

2. เข้าไปยัง directory ของ project
```
cd cs-ai-group-project
```

3. สร้างสภาพแวดล้อม
```
py -m venv env
```

4. เปิดใช้งานตัวสภาพแวดล้อมที่สร้างขึ้น
```
env/Scripts/activate
```

5. ติดตั้ง liaries
```
pip install -r requirements.txt
```

6. รันตัว regression models
```
py main.py
```

---

### สมาชิกในกลุ่ม

1. นาย พันธุ์ธัช สุวรรณวัฒนะ รหัสนิสิต 6630250281
2. นาย ปุณณภพ มีฤทธิ์ รหัสนิสิต 6630250591
3. นาย ปัณณวัฒน์ นิ่งเจริญ รหัสนิสิต 6630250231
4. นาย อติรุจ วัฒนะมงคล รหัสนิสิต 6630250508
5. นาย วรินทร์ สายปัญญา รหัสนิสิต 6630250435
6. นางสาว อัมพุชินี บุญรักษ์ รหัสนิสิต 6630250532