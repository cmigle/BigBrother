// BigBrother — Frontend JavaScript
// Shared utilities and common functionality

(function() {
    'use strict';

    // Add active state visual feedback
    document.querySelectorAll('.nav-link').forEach(link => {
        if (link.href === window.location.href) {
            link.classList.add('active');
        }
    });

    // Console branding
    console.log(
        '%c BIGBROTHER %c — ALWAYS WATCHING ',
        'background:#FF3D00; color:#0A0A0A; font-family:"Inter Tight",sans-serif; font-size:14px; font-weight:800; letter-spacing:3px; padding:6px 10px;',
        'background:#0A0A0A; color:#737373; font-family:"JetBrains Mono",monospace; font-size:11px; letter-spacing:2px; padding:6px 10px; border:1px solid #262626;'
    );

})();
