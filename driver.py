from model import make_model, train
from plot import visualize_all_curves

if __name__ == "__main__":
    V = 100+1

    print('Creating models...')

    sa_model = make_model(V, V, N=2, encoding_mode='sinusoidal', combining_mode='add', max_wavelength=10)
    la_model = make_model(V, V, N=2, encoding_mode='learnable', combining_mode='add', max_wavelength=10)
    sc_model = make_model(V, V, N=2, encoding_mode='sinusoidal', combining_mode='concat', max_wavelength=10)
    lc_model = make_model(V, V, N=2, encoding_mode='learnable', combining_mode='concat', max_wavelength=10)

    print('Starting Training...')
    sc_train_loss, sc_val_loss = train(sc_model)
    lc_train_loss, lc_val_loss = train(lc_model)
    sa_train_loss, sa_val_loss = train(sa_model)
    la_train_loss, la_val_loss = train(la_model)

    visualize_all_curves(sa_train_loss, sa_val_loss, sc_train_loss, sc_val_loss, "sinusoidal addition/concat")
    visualize_all_curves(la_train_loss, la_val_loss, lc_train_loss, lc_val_loss, "learnable addition/concat")

    sa_model = make_model(V, V, N=2, encoding_mode='sinusoidal', combining_mode='add')
    la_model = make_model(V, V, N=2, encoding_mode='learnable', combining_mode='add')
    sc_model = make_model(V, V, N=2, encoding_mode='sinusoidal', combining_mode='concat')
    lc_model = make_model(V, V, N=2, encoding_mode='learnable', combining_mode='concat')