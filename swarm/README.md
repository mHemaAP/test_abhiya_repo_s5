# Deploy in Swarm

```
docker swarm init
```

```
docker stack deploy -c docker-compose.yml mnist_server
```

```
docker stack rm mnist_server
```