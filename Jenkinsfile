pipeline {
    agent any

    environment {
        DOCKERHUB_CREDENTIALS = credentials('9d91b5d4-cd9a-41a2-9c8d-fc3e1434bf23')   
        DOCKERHUB_REPO = 'khangpt/crypto-fastapi'           
        IMAGE_NAME = 'crypto-fastapi'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    sh "docker build -t $DOCKERHUB_REPO:$BUILD_NUMBER ."
                }
            }
        }

        stage('Login to DockerHub') {
            steps {
                script {
                    sh "echo $DOCKERHUB_CREDENTIALS_PSW | docker login -u $DOCKERHUB_CREDENTIALS_USR --password-stdin"
                }
            }
        }

        stage('Push Image') {
            steps {
                script {
                    sh "docker push $DOCKERHUB_REPO:$BUILD_NUMBER"
                    // latest tag nếu muốn
                    sh "docker tag $DOCKERHUB_REPO:$BUILD_NUMBER $DOCKERHUB_REPO:latest"
                    sh "docker push $DOCKERHUB_REPO:latest"
                }
            }
        }
    }
}
