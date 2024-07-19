def loadRFI(rfifile):
    with open(rfifile, 'rb') as fh:
        BM = loadFullbitmap(fh, nBlock, blocklen=packBlock, meta=meta)
        bitmap = BM[pack0:pack0+nPack]
        rfi_tick, rfi_spec1 = loadSpec(fh, pack0, nPack, order_off=order_off, bitmap=bitmap, 
                                       verbose=verbose, bitwidth=bitwidth, hdver=hdver, meta=meta, nBlock=nBlock)
    #get covariance
    rfi_Cov1c, rfi_norm1c = makeCov(rfi_spec1.transpose((1,0,2)), coeff=True)
    rfi_nspec1 = rfi_spec1/rfi_norm1c
    rfi_W1c, V1c = Cov2Eig(rfi_Cov1c)
    
    ## remove the leading eigenmode (last mode)
    V1c0 = V1c[:,:,-1]
    
    return rfi_norm1c, V1c0
    
    
def remoRFI(tmpspec, rfi_norm1c, V1c0):
    #normalization
    nspec2 = tmpspec/rfi_norm1c
    #nulling
    rspec2 = np.zeros_like(nspec2)
    for ch in range(nChan):
        rspec2[:,:,ch] = nspec2[:,:,ch] - (np.tensordot(V1c0[ch].conjugate(), nspec2[:,:,ch], 
                                                        axes=(0,1)).reshape((-1,1))*V1c0[ch].reshape((1,nAnt)))
    
    return rspec2, W2C, W2cr
    
def plotRFI(tmpspec, rspec2, nChan, W2C, W2cr) -> None:
    #make covariance
    #before
    Cov2c, norm2c = makeCov(tmpspec.transpose((1,0,2)), coeff=True)
    W2c, V2c = Cov2Eig(Cov2c)
    #after
    Cov2cr, norm2cr = makeCov(rspec2.transpose((1,0,2)), coeff=True)
    nrspec2 = rspec2/norm2cr                                        
    W2cr, V2cr = Cov2Eig(Cov2cr)
            
    #Plot results
    chan = np.arange(nChan)
    fig, sub = plt.subplots(4,1,figsize=(50, 30))

    ax = sub[0] #power before
    ax.set_ylabel('norm.power (dB)')
    for i in range(nAnt):
        ax.plot(chan, 20*np.log10(np.ma.abs(tmpspec[i]).mean(axis=0)))
    
    ax = sub[1] #eigenmode before
    ax.set_xlabel('Chan')
    ax.set_ylabel('eigenvalue (dB)')
    for i in range(nAnt):
        ax.plot(chan, 20*np.log10(W2c[:,i]))
    
    ax = sub[2] #power after
    ax.set_ylabel('norm.power (dB)')
    for i in range(nAnt):
        ax.plot(chan, 20*np.log10(np.ma.abs(rspec2[i]).mean(axis=0)))
    
    ax = sub[3] #eigenmode after
    ax.set_xlabel('Chan')
    ax.set_ylabel('eigenvalue (dB)')
    for i in range(nAnt):
        ax.plot(chan, 20*np.log10(W2cr[:,i]))
    tag = 't%07d' % dt
    fig.savefig('%s/rfirem_res_%s.png' % (cdir, tag))
    plt.close(fig)
