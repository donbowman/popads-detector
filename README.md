# popads-detector

Popads has been using a [domain generation algorithm](https://en.wikipedia.org/wiki/Domain_generation_algorithm)
to circumvent ad-blockers (which operate on regex and well known domains).

In turn, crypto-miner malware has been
[delivered](http://blog.netlab.360.com/who-is-stealing-my-power-iii-an-adnetwork-company-case-study-en/)
via ads on this network, 'cryptojacking'

An example of the domains used can be found on
[github](https://raw.githubusercontent.com/Yhonay/antipopads/master/hosts)

```
aaeqlxdgx.bid
aajychvi.bid
aanvxbvkdxph.com
aaomstbnbiqo.com
 ...
```

making these difficult to block in conventional ways.

This code teaches a [LSTM neural network](https://en.wikipedia.org/wiki/Long_short-term_memory)
to detect these host names. It does so by training against the
[Cisco Umbrella 1 million](https://umbrella.cisco.com/blog/2016/12/14/cisco-umbrella-1-million/)
set as a 'good' set, and against a set from [Yhonay github rebpo](https://github.com/Yhonay/antipopads)
as the 'bad'.

The popads list is split in half, half is used for training, and the other half to
test the efficacy.

More info [on my blog](https://blog.donbowman.ca/2018/03/04/recognising-popunder-advertisements-with-machine-learning-an-implementation/)

## License

Copyright 2018 Don Bowman <db@donbowman.ca>

Licensed under the Apache License, Version 2.0


