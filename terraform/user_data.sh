#!/bin/bash

aws secretsmanager get-secret-value --secret-id github_credentials --region us-east-1 --query 'SecretString' --output text > /home/ec2-user/.ssh/id_ed25519
chown ec2-user /home/ec2-user/.ssh/id_ed25519
chgrp ec2-user /home/ec2-user/.ssh/id_ed25519 
chmod 0600 /home/ec2-user/.ssh/id_ed25519

#Add github's authorized_key
touch /home/ec2-user/.ssh/known_hosts
if ! grep -q "^github.com" /home/ec2-user/.ssh/known_hosts; then
     ssh-keyscan -t rsa -p 443 ssh.github.com >> /home/ec2-user/.ssh/known_hosts
fi
cat > /home/ec2-user/.ssh/config << EOT
Host github.com
Hostname ssh.github.com
Port 443
User git
EOT
chown ec2-user /home/ec2-user/.ssh/known_hosts
chgrp ec2-user /home/ec2-user/.ssh/known_hosts 
chown ec2-user /home/ec2-user/.ssh/config
chgrp ec2-user /home/ec2-user/.ssh/config 
chmod 0600 /home/ec2-user/.ssh/config
chmod 0600 /home/ec2-user/.ssh/known_hosts

#Clone the repo
sudo -u ec2-user git clone ssh://github.com/odemeulder/nlptransformers /home/ec2-user/nlptransformers
su ec2-user -c 'pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html'
su ec2-user -c 'pip install -r /home/ec2-user/nlptransformers/requirements.txt'

