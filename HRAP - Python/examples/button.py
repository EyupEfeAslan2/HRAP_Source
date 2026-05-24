from matplotlib.widgets import Button

def attach_hd_buttons(fig, axs_list, callback_fn):
    """
    Attaches an 'HD' button to the bottom-right inside of each axis in axs_list.
    Returns the list of button objects (must be kept in memory to prevent garbage collection).
    """
    buttons = []
    for idx, ax in enumerate(axs_list):
        # ax.inset_axes uses coordinates relative to the subplot itself (0 to 1).
        # [x0, y0, width, height] -> Placing it at the bottom-right corner INSIDE the plot
        btn_ax = ax.inset_axes([0.83, 0.03, 0.15, 0.14])
        
        # Using "HD" text instead of an unsupported Unicode character
        btn = Button(btn_ax, '+', color='#f0f0f0', hovercolor='#cccccc')
        btn_ax.set_zorder(10)
        
        # Make the button background slightly transparent so it doesn't completely block data lines
        btn_ax.patch.set_alpha(0.85)
        
        # Make the button border less aggressive
        for spine in btn_ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('#888888')
            
        # Adjust text styling to fit the small button
        btn.label.set_fontsize(9)  
        btn.label.set_fontweight('bold')
        btn.label.set_color('#333333')
        
        # Closure capture via default arg
        btn.on_clicked(lambda event, i=idx: callback_fn(i))
        buttons.append(btn)
        
    return buttons