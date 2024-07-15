import os
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import datetime
import pytz
def sendgmail(receivermailid,first_name):
    with open('intrusion.jpg', 'rb') as f:
        img_data = f.read()
    #img_data=ImgFileName
    #print(img_data)

    msg = MIMEMultipart()
    msg['Subject'] = 'subject'
    msg['From'] = 'e@mail.cc'
    msg['To'] = 'e@mail.cc'
    text1=MIMEText("INTRUSION DETECTED")
    timevar=datetime.datetime.now(pytz.timezone('Asia/Kolkata'))
    temp="""<!DOCTYPE html><html lang="en">
    <head>
    <meta charset="UTF-8">
    <title>Email Template</title>
    <style>
        /* Add your styles here */
        body {
            background-color: #F5F5F5;
            font-family: Arial, sans-serif;
            font-size: 16px;
            line-height: 1.5;
            color: #333;
            padding: 0;
            margin: 0;
        }
        table {
            width: 100%;
            max-width: 600px;
            margin: 0 auto;
            background-color: #FFF;
            border-collapse: collapse;
            border-spacing: 0;
        }
        td {
            padding: 20px;
        }
        h1 {
            margin-top: 0;
            margin-bottom: 20px;
            font-size: 28px;
            line-height: 1.2;
            color: #333;
        }
        p {
            margin-top: 0;
            margin-bottom: 20px;
            font-size: 40px;
            line-height: 1.5;
            color: #333;
        }
        .button {
            display: inline-block;
            margin-top: 20px;
            padding: 10px 20px;
            background-color: #333;
            color: #FFF;
            text-decoration: none;
            border-radius: 5px;
        }
         
    </style>
</head>
<body>
    <table>
        <tr>
            <td>
                <!-- Header -->
                <h1 style="color: #FFF; background-color: #333; padding: 20px;">"""+ first_name+"""   ALERT !!!</h1>
   <img src="https://cdn-icons-png.flaticon.com/512/189/189484.png?w=740&t=st=1682321869~exp=1682322469~hmac= 5d4a228d5e55099c675ca8d149aeb782665fcdb43f5ec3a1688f924c779099bb" alt="Thief Image">
            </td>
        </tr>
        <tr>
            <td>
                <!-- Body -->
                <p>                            Intrusion detected</p>
                
                <p>INTRUSION TIME:"""+str(timevar)+""" </p>
            </td>
        </tr>
    </table>
</body>
</html>"""
    text = MIMEText(temp, 'html')
    msg.attach(text)
    print("sent         dkveoihggroi vdkkvoii")
    image = MIMEImage(img_data)
    image.add_header('Content-Disposition', 'attachment', filename="intrusion.jpg")
    msg.attach(image)
    server=smtplib.SMTP('smtp.gmail.com',587)
    server.ehlo()
    server.starttls()
    server.login('thyag7363@gmail.com','xpzxofxizukpiowb')
    server.sendmail('thyag7363@gmail.com',receivermailid,msg.as_string())
    server.quit()
    #.format(first_name, timevar)