https://docs.microsoft.com/en-us/azure/container-instances/container-instances-tutorial-prepare-app

1. login using the command 'az login' and follow the interactive sign-in procedure
2. then get the registry name from the main dashboard. It will look like this ->"xcdtest281215d679b.azurecr.io".
   Remove the **.azurecr.io** in the end and use the remaining in the following command 
   'az acr login --name xcdtest281215d679b'
3. Next push image to registry **BUT**  you must tag it with the fully qualified name of your ACR login server. The login server name is in the format
   xcdtest281215d679b.azurecr.io (all lowercase) i.e include the azurecr.io which was removed in the previous part. Now push using
   
    docker pull hello-world    # pull from docker-hub
    docker tag hello-world <acrLoginServer>/hello-world:v1 # where acrLoginServer="xcdtest281215d679b.azurecr.io"
    docker push <acrLoginServer>/hello-world:v1
    
4. Check the repository in the azure container
    

