import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(sender, recipient, subject, body):
    # Create a multipart message
    message = MIMEMultipart()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = subject

    # Add body to the email
    message.attach(MIMEText(body, "plain"))

    # Connect to the Amazon SES SMTP server
    with smtplib.SMTP("email-smtp.us-east-2.amazonaws.com", 587) as server:
        # Start TLS encryption
        server.starttls()
        # Login to your Amazon SES account
        server.login("AKIAVZ6AJ7TPPP2NHFNO", "BBG9nAp5/WzQNOBI6xrs3H5zyeOI36udx6y32T4jcqQA")
        # Send the email
        server.sendmail(sender, recipient, message.as_string())

# Example usage
sender_email = "prangineni@hawk.iit.edu"
recipient_email = "pavanirangineni123@gmail.com"
subject = "Hello from Python"
body = "This is the body of the email."

send_email(sender_email, recipient_email, subject, body)
