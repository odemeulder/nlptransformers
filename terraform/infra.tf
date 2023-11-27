provider "aws" {
  profile    = "default"
  region     = "us-east-1"
}

resource "aws_iam_role" "dl_role" {
  name = "odm_secrets_reader"
  path = "/"
  managed_policy_arns = [
    "arn:aws:iam::572028816325:policy/OdmSecretsManagerAccessPolicy",
    "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"    
  ]
  assume_role_policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Action": "sts:AssumeRole",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Effect": "Allow",
      "Sid": ""
    }
  ]
}
EOF

  tags = {
    Application = "dl"
  }
}

resource "aws_iam_instance_profile" "dl_profile" {
  role = "${aws_iam_role.dl_role.name}"
}

resource "aws_security_group" "dl_sg" {
  vpc_id = "vpc-edf80c94"

  ingress {
    description = "SSH from everywhere"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Application = "dl"
  }
}

resource "aws_instance" "dl" {
  ami           = "ami-03a3697fb8fa1cc70"
  instance_type = "p3.2xlarge"
  iam_instance_profile = "${aws_iam_instance_profile.dl_profile.name}"
  key_name = "olivier-ed"
  vpc_security_group_ids = [ "${aws_security_group.dl_sg.id}" ]
  user_data = base64encode(file("${path.module}/user_data.sh"))

  tags = {
    Application = "dl"
    Environment = "prd"
  }
}

output "instance_ips" {
  value = aws_instance.dl.*.public_ip
}

output "instance_sg_id" {
  value = aws_security_group.dl_sg.id
}
